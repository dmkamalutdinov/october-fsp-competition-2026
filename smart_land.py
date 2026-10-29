#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_land.py - мягкая посадка Pioneer Mini на любую поверхность.
Использование из своего кода:
    from smart_land import smart_land
    smart_land(drone)

Самостоятельный тест (взлетит, повисит, сядет):
    python smart_land.py --demo --height 1.0
Если дрон уже в воздухе и управляется другим скриптом - вызывайте smart_land(drone) оттуда.
"""
import argparse
import statistics
import time

# ---------------- настройки (можно править) ----------------
CUT_HEIGHT = 0.12        # м, на этой высоте над поверхностью выключаем моторы
V_MAX = 0.30             # м/с, максимальная скорость снижения
V_MIN = 0.10             # м/с, минимальная (у поверхности)
SLOW_ZONE = 0.5          # м, ниже этой высоты начинаем замедляться
RATE_HZ = 10.0           # частота отправки set_manual_speed (метод нужно слать в цикле)
STABLE_WINDOW = 1.5      # с, окно оценки стабильности
STABLE_HEIGHT_TOL = 0.04  # м, допустимый разброс высоты в окне
STABLE_POS_TOL = 0.06    # м, допустимый разброс x/y в окне
STABLE_TIMEOUT = 10.0    # с, дольше не ждём стабилизацию
MAX_JUMP = 0.30          # м, скачок высоты между соседними замерами = помеха (flow/дальномер)
INVALID_LIMIT = 1.0      # с, сколько терпим отсутствие корректной высоты
TOTAL_TIMEOUT = 40.0     # с, предел на всю посадку (потом обычный land())
LOW_SAFE = 0.35          # м, ниже этого при проблемах сразу disarm


def _height(drone):
    """Высота по дальномеру (м) или None, если данных нет / невалидны."""
    try:
        h = drone.get_dist_sensor_data(get_last_received=True)
    except Exception:
        return None
    if h is None:
        return None
    try:
        h = float(h)
    except (TypeError, ValueError):
        return None
    if h != h or h <= 0.0 or h > 10.0:
        return None
    return h


def _xy(drone):
    try:
        p = drone.get_local_position_lps(get_last_received=True)
        if p is not None and len(p) >= 2:
            return float(p[0]), float(p[1])
    except Exception:
        pass
    return None


def wait_stable(drone, log=print):
    """Ждём, пока дрон перестанет качаться: разброс высоты и позиции в окне мал."""
    t0 = time.time()
    hist = []
    dt = 1.0 / RATE_HZ
    while time.time() - t0 < STABLE_TIMEOUT:
        now = time.time()
        h, xy = _height(drone), _xy(drone)
        if h is not None:
            hist.append((now, h, xy))
        hist = [s for s in hist if now - s[0] <= STABLE_WINDOW]
        if len(hist) >= int(STABLE_WINDOW * RATE_HZ * 0.6) and now - t0 >= STABLE_WINDOW:
            hs = [s[1] for s in hist]
            ok = max(hs) - min(hs) <= STABLE_HEIGHT_TOL
            xs = [s[2] for s in hist if s[2] is not None]
            if xs:
                ok = ok and (max(p[0] for p in xs) - min(p[0] for p in xs) <= STABLE_POS_TOL) \
                        and (max(p[1] for p in xs) - min(p[1] for p in xs) <= STABLE_POS_TOL)
            if ok:
                log("[land] дрон стабилен, высота %.2f м" % statistics.median(hs))
                return True
        time.sleep(dt)
    log("[land] стабилизации не дождались за %.0f с, продолжаем" % STABLE_TIMEOUT)
    return False


def _det_down_sign(drone, log=print):
    """Определяем знак vz, при котором высота по дальномеру уменьшается (на случай разных соглашений осей)."""
    h0 = _median_height(drone, 5)
    if h0 is None:
        return -1.0
    t_end = time.time() + 0.8
    while time.time() < t_end:
        drone.set_manual_speed(0.0, 0.0, -0.15, 0.0)
        time.sleep(1.0 / RATE_HZ)
    h1 = _median_height(drone, 3, send_zero=True)
    if h1 is None:
        return -1.0
    if h1 > h0 + 0.02:
        log("[land] vz инвертирован: для снижения используем +vz")
        return 1.0
    return -1.0


def _median_height(drone, n, send_zero=False):
    vals = []
    for _ in range(n):
        if send_zero:
            drone.set_manual_speed(0.0, 0.0, 0.0, 0.0)
        h = _height(drone)
        if h is not None:
            vals.append(h)
        time.sleep(1.0 / RATE_HZ)
    return statistics.median(vals) if vals else None


def smart_land(drone, cut_height=CUT_HEIGHT, use_land_at_end=False, log=print):
    """Мягкая посадка. Возвращает True, если моторы выключены на низкой высоте."""
    t_start = time.time()
    dt = 1.0 / RATE_HZ
    if _height(drone) is None and _median_height(drone, 10) is None:
        log("[land] нет данных дальномера - обычный land()")
        drone.land()
        return False

    wait_stable(drone, log)
    sign = _det_down_sign(drone, log)

    last_h = _median_height(drone, 3, send_zero=True)
    last_valid_t = time.time()
    recent = []
    while time.time() - t_start < TOTAL_TIMEOUT:
        h = _height(drone)
        now = time.time()
        if h is None:
            if now - last_valid_t > INVALID_LIMIT:
                if last_h is not None and last_h < LOW_SAFE:
                    log("[land] дальномер пропал у поверхности (%.2f м) - disarm" % last_h)
                    drone.disarm()
                    return True
                drone.set_manual_speed(0.0, 0.0, 0.0, 0.0)   # зависаем и ждём данные
            time.sleep(dt)
            continue
        if last_h is not None and abs(h - last_h) > MAX_JUMP:
            # выброс (например, край куба под дальномером или сбой flow): пропускаем замер
            log("[land] скачок высоты %.2f -> %.2f, замер пропущен" % (last_h, h))
            drone.set_manual_speed(0.0, 0.0, 0.0, 0.0)
            last_valid_t = now
            if last_h < LOW_SAFE and h < LOW_SAFE:
                log("[land] нестабильно у поверхности - disarm")
                drone.disarm()
                return True
            time.sleep(dt)
            last_h = h if abs(h - last_h) > 2 * MAX_JUMP else last_h
            continue
        last_h, last_valid_t = h, now
        recent = (recent + [h])[-3:]
        hm = statistics.median(recent)
        if hm <= cut_height:
            log("[land] высота %.2f м <= %.2f - выключаем моторы" % (hm, cut_height))
            drone.set_manual_speed(0.0, 0.0, 0.0, 0.0)
            if use_land_at_end:
                drone.land()
            else:
                drone.disarm()
            return True
        k = min(1.0, max(0.0, (hm - cut_height) / max(SLOW_ZONE - cut_height, 1e-6)))
        v = V_MIN + (V_MAX - V_MIN) * k
        drone.set_manual_speed(0.0, 0.0, sign * v, 0.0)
        time.sleep(dt)

    log("[land] таймаут посадки - обычный land()")
    drone.land()
    return False


def main():
    ap = argparse.ArgumentParser(description="Демонстрация мягкой посадки Pioneer Mini")
    ap.add_argument("--demo", action="store_true", help="взлететь, повисеть и сесть (иначе считаем, что дрон уже в воздухе)")
    ap.add_argument("--height", type=float, default=1.0, help="высота зависания в демо, м")
    ap.add_argument("--cut", type=float, default=CUT_HEIGHT, help="высота выключения моторов, м")
    ap.add_argument("--simulator", action="store_true")
    args = ap.parse_args()

    from pioneer_sdk import Pioneer
    drone = Pioneer(simulator=True) if args.simulator else Pioneer()
    time.sleep(3)
    try:
        if args.demo:
            drone.arm()
            time.sleep(0.5)
            drone.takeoff()
            time.sleep(4)
            loc = drone.get_local_position_lps(True) or [0, 0, 0]
            drone.go_to_local_point(loc[0], loc[1], args.height, 0)
            time.sleep(4)
        smart_land(drone, cut_height=args.cut)
    except KeyboardInterrupt:
        print("\nCtrl+C: аварийное выключение моторов")
        drone.disarm()
    finally:
        time.sleep(1)
        drone.close_connection()


if __name__ == "__main__":
    main()
