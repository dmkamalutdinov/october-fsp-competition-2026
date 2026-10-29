#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pioneer_aruco_nav.py - навигация Pioneer Mini по полю ArUco-маркеров.

Вычисления выполняются на ноутбуке (дрон = hotspot ESP32, onboard computing нет).
Камера смотрит ВНИЗ. Алгоритм:

  1. Загружаем карту маркеров из txt (формат Clover: id length x y z rot_z rot_y rot_x).
  2. По кадру ищем маркеры, делаем solvePnP по всем найденным углам сразу
     -> положение камеры (дрона) в системе координат КАРТЫ.
  3. Один раз после взлёта калибруем связь "локальные координаты Pioneer -> координаты карты":
     двигаем дрон на 0.4 м по local X и по local Y, смотрим на сколько сместился по карте
     и получаем матрицу 2x2. Так не нужно угадывать ни направление осей Pioneer,
     ни поворот/отражение картинки камеры.
  4. Цикл: спрашиваем ID маркера -> по камере узнаём позицию -> считаем ошибку в карте ->
     переводим в локальные координаты -> go_to_local_point -> снова смотрим камерой
     -> повторяем, пока не окажемся в допуске.

Картинки могут приходить с задержкой или теряться: кадры берутся с повторами, дубликаты
отбрасываются, после остановки кадры "сливаются" (drain), поза берётся по медиане окна.

Режимы без полёта:
  --pose-only         печатать позицию камеры в карте (двигайте дрон рукой, проверьте оси)
  --find-dict         подобрать ArUco-словарь по живому кадру
  --image foto.png    оценить позу по файлу
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# =============================================================================
#                              НАСТРОЙКИ
# =============================================================================
DEFAULT_MAP = "fsp.txt"                 # файл карты (можно задать --map)
DEFAULT_CALIB = "camera_calib.npz"      # результат calibrate_camera.py

ARUCO_DICT = "DICT_4X4_250"             # ПРОВЕРЬТЕ через --find-dict
FLIGHT_HEIGHT = 1.5                     # м, фиксированная высота полёта
TAKEOFF_WAIT = 4.0                      # с, ожидание после takeoff()

# Если файла калибровки нет, используем грубую модель без дисторсии.
FALLBACK_HFOV_DEG = 80.0                # горизонтальный угол обзора камеры (подберите)

# Смещение камеры относительно центра дрона в системе камеры, м:
# x - вправо по картинке, y - вниз по картинке, z - вперёд (от объектива).
CAMERA_OFFSET_CAM = (0.0, 0.0, 0.0)

# --- распознавание / поза ---
MIN_MARKERS = 2                         # минимум маркеров для оценки позы
MAX_REPROJ_PX = 4.0                     # отбрасывать позу при большей ошибке репроекции
POSE_WINDOW = 3                         # размер окна кадров для медианы
POSE_STABLE_TOL = 0.04                  # м, допустимый разброс позы внутри окна
POSE_TIMEOUT = 8.0                      # с, сколько ждать валидную позу
FRAME_DRAIN_TIME = 1.0                  # с, выбрасывать кадры после остановки (лаг сети)

# --- полёт ---
TOLERANCE = 0.08                        # м, допуск до центра маркера
MAX_ITERATIONS = 6                      # максимум итераций коррекции на один маркер
MAX_STEP = 1.0                          # м, максимальный шаг за одну команду
SETTLE_TIME = 1.2                       # с, пауза после достижения точки
MOVE_TIMEOUT = 15.0                     # с, максимум ожидания point_reached
PROBE_DISTANCE = 0.4                    # м, шаг калибровочных смещений local->map
AXES_FIXED_DEG = None                   # угол оси local X в карте (град); None = калибровать при взлёте
AXES_MIRROR = False                     # True, если оси зеркальны (из вывода калибровки)
CAL_MAX_SCALE_ERR = 0.30                # допустимое отклонение масштаба при калибровке осей


# =============================================================================
#                              КАРТА МАРКЕРОВ
# =============================================================================
class Marker:
    __slots__ = ("id", "length", "center", "rot", "corners")

    def __init__(self, mid, length, center, rot):
        self.id = mid
        self.length = length
        self.center = center
        self.rot = rot
        h = length / 2.0
        # порядок углов OpenCV: левый-верх, правый-верх, правый-низ, левый-низ
        # (маркер лежит на карте "правильно": верх маркера = +y карты, право = +x)
        local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        self.corners = (rot @ local.T).T + center


def _rot_matrix(rz, ry, rx):
    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def load_map(path):
    """Читает любой txt в формате: id length x y z [rot_z rot_y rot_x].
    Разделители - пробелы/табы/запятые, строки с # игнорируются."""
    markers = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(";", " ").replace(",", " ").split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                raise ValueError("%s:%d: не удалось разобрать строку: %r" % (path, n, line))
            if len(vals) < 5:
                raise ValueError("%s:%d: нужно минимум 5 значений (id length x y z)" % (path, n))
            vals += [0.0] * (8 - len(vals))
            mid = int(vals[0])
            center = np.array(vals[2:5], dtype=np.float64)
            rot = _rot_matrix(vals[5], vals[6], vals[7])
            markers[mid] = Marker(mid, vals[1], center, rot)
    if not markers:
        raise ValueError("Карта %s пуста" % path)
    return markers


# =============================================================================
#                              КАМЕРА / ПОЗА
# =============================================================================
class CameraModel:
    def __init__(self, calib_path=None):
        self.K = None
        self.dist = np.zeros(5)
        self.size = None
        self.calibrated = False
        if calib_path and os.path.exists(calib_path):
            d = np.load(calib_path)
            self.K = d["K"].astype(np.float64)
            self.dist = d["dist"].astype(np.float64).ravel()
            self.size = tuple(int(v) for v in d["image_size"])  # (w, h)
            self.calibrated = True
            print("[cam] калибровка загружена: %s" % calib_path)
        else:
            print("[cam] ВНИМАНИЕ: файл калибровки не найден, используется грубая модель "
                  "(HFOV=%.0f°, без дисторсии). Точность будет хуже." % FALLBACK_HFOV_DEG)

    def matrices(self, shape):
        h, w = shape[:2]
        if self.calibrated:
            K = self.K.copy()
            cw, ch = self.size
            if (w, h) != (cw, ch):
                K[0, :] *= w / cw
                K[1, :] *= h / ch
            return K, self.dist
        f = (w / 2.0) / math.tan(math.radians(FALLBACK_HFOV_DEG) / 2.0)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        return K, self.dist


def make_detector(dict_name):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError("Неизвестный словарь ArUco: %s" % dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):                 # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda img: det.detectMarkers(img)[:2]
    params = cv2.aruco.DetectorParameters_create()          # старые версии
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)[:2]


class Pose:
    """Положение дрона в системе карты."""
    def __init__(self, xyz, n_markers, reproj, R):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.n = n_markers
        self.reproj = reproj
        self.R = R   # поворот map->camera

    @property
    def xy(self):
        return self.xyz[:2]


def estimate_pose(frame, detect, markers, cam, debug_path=None):
    """Возвращает Pose или None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids = detect(gray)
    if ids is None or len(ids) == 0:
        return None
    obj, img, used = [], [], []
    for c, i in zip(corners, ids.flatten()):
        m = markers.get(int(i))
        if m is None:
            continue
        obj.append(m.corners)
        img.append(c.reshape(4, 2))
        used.append(int(i))
    if len(used) < MIN_MARKERS:
        return None
    obj = np.concatenate(obj).astype(np.float64)
    img = np.concatenate(img).astype(np.float64)
    K, dist = cam.matrices(gray.shape)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
    if reproj > MAX_REPROJ_PX:
        return None
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).ravel()
    pos = cam_pos + R.T @ np.array(CAMERA_OFFSET_CAM)
    if debug_path:
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.putText(vis, "x=%.2f y=%.2f z=%.2f n=%d err=%.1fpx" % (pos[0], pos[1], pos[2], len(used), reproj),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(debug_path, vis)
    return Pose(pos, len(used), reproj, R)


class PoseSource:
    """Получает кадры с дрона (с повторами) и считает позу в карте."""
    def __init__(self, camera, detect, markers, cam_model, debug_dir=None):
        self.camera = camera
        self.detect = detect
        self.markers = markers
        self.cam = cam_model
        self.debug_dir = debug_dir
        self._last = None
        self._cnt = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def _frame(self):
        """Один НОВЫЙ кадр или None (дубликаты отбрасываем)."""
        try:
            f = self.camera.get_cv_frame()
        except Exception as e:                 # обрыв сети и т.п.
            print("[cam] ошибка получения кадра: %s" % e)
            time.sleep(0.2)
            return None
        if f is None:
            return None
        if self._last is not None and f.shape == self._last.shape and np.array_equal(f, self._last):
            return None
        self._last = f
        return f

    def drain(self, seconds):
        """Выбрасываем накопившиеся (устаревшие) кадры после движения."""
        t_end = time.time() + seconds
        while time.time() < t_end:
            if self._frame() is None:
                time.sleep(0.03)

    def get_pose(self, timeout=POSE_TIMEOUT, window=POSE_WINDOW):
        """Медиана позы по окну свежих кадров. None, если не получилось."""
        t_end = time.time() + timeout
        buf = deque(maxlen=window)
        last_good = None
        while time.time() < t_end:
            f = self._frame()
            if f is None:
                time.sleep(0.03)
                continue
            dbg = None
            if self.debug_dir:
                self._cnt += 1
                dbg = os.path.join(self.debug_dir, "f%05d.jpg" % self._cnt)
            p = estimate_pose(f, self.detect, self.markers, self.cam, dbg)
            if p is None:
                buf.clear()          # потеряли карту - начинаем окно заново
                continue
            buf.append(p)
            last_good = p
            if len(buf) == window:
                arr = np.array([q.xyz for q in buf])
                spread = float(np.max(np.ptp(arr[:, :2], axis=0)))
                if spread <= POSE_STABLE_TOL:
                    med = np.median(arr, axis=0)
                    return Pose(med, int(np.median([q.n for q in buf])),
                                float(np.mean([q.reproj for q in buf])), buf[-1].R)
        if last_good is not None and len(buf) >= 2:
            arr = np.array([q.xyz for q in buf])
            print("[pose] окно нестабильно (разброс > %.0f см), беру медиану как есть" % (POSE_STABLE_TOL * 100))
            return Pose(np.median(arr, axis=0), last_good.n, last_good.reproj, last_good.R)
        return None


# =============================================================================
#                              ДРОН
# =============================================================================
class DroneNav:
    def __init__(self, drone, poses, markers):
        self.drone = drone
        self.poses = poses
        self.markers = markers
        self.A = None          # local(2) -> map(2)
        self.A_inv = None
        self.yaw = 0.0
        self.last_local = None  # последняя известная локальная позиция [x, y]

    # ---- вспомогательное ----
    def local_xy(self):
        p = None
        try:
            p = self.drone.get_local_position_lps(True)
        except Exception:
            pass
        if p is not None and len(p) >= 2 and all(v is not None for v in p[:2]):
            self.last_local = np.array(p[:2], dtype=np.float64)
        return self.last_local

    def goto_local(self, x, y, wait=True):
        self.drone.go_to_local_point(float(x), float(y), FLIGHT_HEIGHT, self.yaw)
        if not wait:
            return
        time.sleep(0.5)    # даём флагу point_reached сброситься
        t_end = time.time() + MOVE_TIMEOUT
        while time.time() < t_end:
            if self.drone.point_reached():
                break
            time.sleep(0.1)
        else:
            print("[fly] point_reached не сработал за %.0f с, продолжаю" % MOVE_TIMEOUT)
        time.sleep(SETTLE_TIME)
        self.poses.drain(FRAME_DRAIN_TIME)

    # ---- взлёт/посадка ----
    def takeoff(self):
        try:
            yaw = self.drone.get_yaw(True)
            self.yaw = float(yaw) if yaw is not None else 0.0
        except Exception:
            self.yaw = 0.0
        self.drone.arm()
        time.sleep(0.5)
        self.drone.takeoff()
        time.sleep(TAKEOFF_WAIT)
        loc = self.local_xy()
        if loc is None:
            loc = np.zeros(2)
            self.last_local = loc
        self.goto_local(loc[0], loc[1])      # выходим на рабочую высоту
        print("[fly] взлёт выполнен, рабочая высота %.2f м" % FLIGHT_HEIGHT)

    def land(self):
        try:
            try:
                from smart_land import smart_land      # мягкая посадка (см. smart_land.py)
                smart_land(self.drone)
                return
            except ImportError:
                pass
            self.drone.land()
        except Exception as e:
            print("[fly] ошибка посадки: %s" % e)

    # ---- калибровка осей local -> map ----
    def calibrate_axes(self):
        if AXES_FIXED_DEG is not None:
            a = math.radians(AXES_FIXED_DEG)
            c, sn = math.cos(a), math.sin(a)
            self.A = np.array([[c, -sn], [sn, c]]) @ np.diag([1.0, -1.0 if AXES_MIRROR else 1.0])
            self.A_inv = self.A.T
            print("[cal] оси заданы вручную: local X под %.1f° в карте%s" %
                  (AXES_FIXED_DEG, ", зеркально" if AXES_MIRROR else ""))
            return True
        print("[cal] калибровка осей: смотрим, как local-смещения отражаются на карте...")
        p0 = self.poses.get_pose()
        l0 = self.local_xy()
        if p0 is None or l0 is None:
            print("[cal] нет позы по камере или локальных координат")
            return False
        dl, dm = [], []
        base_l, base_m = l0.copy(), p0.xy.copy()
        for axis in (0, 1):
            target = base_l.copy()
            target[axis] += PROBE_DISTANCE
            self.goto_local(target[0], target[1])
            p1 = self.poses.get_pose()
            l1 = self.local_xy()
            if p1 is None or l1 is None:
                print("[cal] потеряли позу во время калибровки")
                return False
            dl.append(l1 - base_l)
            dm.append(p1.xy - base_m)
            # возвращаемся в базовую точку
            self.goto_local(base_l[0], base_l[1])
            base_m = (self.poses.get_pose() or p0).xy.copy()
            base_l = (self.local_xy() if self.local_xy() is not None else base_l).copy()
        DL = np.array(dl).T       # столбцы - смещения по local
        DM = np.array(dm).T       # столбцы - смещения по карте
        if abs(np.linalg.det(DL)) < 1e-3:
            print("[cal] дрон не сместился по local (LPS не работает?)")
            return False
        A = DM @ np.linalg.inv(DL)
        U, S, Vt = np.linalg.svd(A)
        print("[cal] сырая матрица local->map:\n%s\n[cal] сингулярные значения (ожидается ~1): %s" % (A, S))
        if np.any(np.abs(S - 1.0) > CAL_MAX_SCALE_ERR):
            print("[cal] ВНИМАНИЕ: масштаб сильно отличается от 1 - проверьте калибровку камеры "
                  "и высоту (FLIGHT_HEIGHT)")
            if np.any(S < 0.2):
                return False
        self.A = U @ Vt           # ортогональная часть: поворот или поворот с отражением
        self.A_inv = self.A.T
        ang = math.degrees(math.atan2(self.A[1, 0], self.A[0, 0]))
        refl = np.linalg.det(self.A) < 0
        print("[cal] OK: ось local X направлена в карте под %.1f°%s" %
              (ang, ", оси ЗЕРКАЛЬНЫ (левая/правая тройка)" if refl else ""))
        return True

    # ---- полёт к маркеру ----
    def fly_to_marker(self, mid):
        m = self.markers[mid]
        target = m.center[:2]
        print("[nav] цель: маркер %d, карта (%.2f, %.2f)" % (mid, target[0], target[1]))
        misses = 0
        for it in range(1, MAX_ITERATIONS + 1):
            pose = self.poses.get_pose()
            if pose is None:
                misses += 1
                print("[nav] карта не видна (%d), вишу и пробую снова" % misses)
                if misses >= 3:
                    print("[nav] карту потерял. Остановка; решите: выберите маркер повторно или land")
                    return False
                continue
            misses = 0
            err = target - pose.xy
            dist = float(np.linalg.norm(err))
            print("[nav] итер %d: поза карта (%.2f, %.2f), z=%.2f, маркеров=%d, ошибка=%.2f м (%.0f см)"
                  % (it, pose.xy[0], pose.xy[1], pose.xyz[2], pose.n, dist, dist * 100))
            if dist <= TOLERANCE:
                print("[nav] ЦЕЛЬ ДОСТИГНУТА (в пределах %.0f см)" % (TOLERANCE * 100))
                return True
            if dist > MAX_STEP:
                err = err / dist * MAX_STEP
            dl = self.A_inv @ err
            cur = self.local_xy()
            if cur is None:
                print("[nav] нет локальной позиции, прерываю")
                return False
            tgt = cur + dl
            print("[nav] команда: local (%.2f, %.2f) -> (%.2f, %.2f), z=%.2f" %
                  (cur[0], cur[1], tgt[0], tgt[1], FLIGHT_HEIGHT))
            self.goto_local(tgt[0], tgt[1])
        pose = self.poses.get_pose()
        if pose is not None:
            d = float(np.linalg.norm(target - pose.xy))
            print("[nav] итерации исчерпаны, остаточная ошибка %.0f см" % (d * 100))
            return d <= TOLERANCE * 2
        return False


# =============================================================================
#                              СЕРВИСНЫЕ РЕЖИМЫ
# =============================================================================
def find_dictionary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    res = []
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_") or "APRILTAG" in name:
            continue
        try:
            ids = make_detector(name)(gray)[1]
        except Exception:
            continue
        res.append((0 if ids is None else len(ids), name, [] if ids is None else sorted(int(i) for i in ids.flatten())))
    res.sort(reverse=True)
    print("Словари по числу найденных маркеров:")
    for n, name, ids in res[:6]:
        print("  %-22s %2d  %s" % (name, n, ids[:12]))
    print("Подходящий - тот, где найдены ожидаемые ID (0..170 для вашей карты).")


def grab_one_frame(camera, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        f = camera.get_cv_frame()
        if f is not None:
            return f
        time.sleep(0.1)
    return None


def prompt_loop(nav, markers):
    ids = sorted(markers)
    print("\nКарта: %d маркеров, ID %d..%d" % (len(ids), ids[0], ids[-1]))
    print("Команды: <ID> - лететь на маркер | p - показать позу | c - перекалибровать оси | l - посадка и выход")
    while True:
        try:
            s = input("\nНа какой маркер лететь? > ").strip().lower()
        except EOFError:
            s = "l"
        if s in ("l", "q", "land", "exit"):
            return
        if s == "p":
            p = nav.poses.get_pose()
            print("поза:", "нет" if p is None else "карта (%.2f, %.2f, %.2f) маркеров=%d" %
                  (p.xyz[0], p.xyz[1], p.xyz[2], p.n))
            continue
        if s == "c":
            nav.calibrate_axes()
            continue
        if not s.lstrip("-").isdigit():
            print("Введите номер маркера или команду")
            continue
        mid = int(s)
        if mid not in markers:
            print("Маркера %d нет в карте" % mid)
            continue
        nav.fly_to_marker(mid)


def main():
    global FLIGHT_HEIGHT
    ap = argparse.ArgumentParser(description="Навигация Pioneer Mini по карте ArUco-маркеров")
    ap.add_argument("--map", default=DEFAULT_MAP, help="txt с картой маркеров")
    ap.add_argument("--calib", default=DEFAULT_CALIB, help="файл калибровки камеры (.npz)")
    ap.add_argument("--dict", default=ARUCO_DICT, help="словарь ArUco, напр. DICT_4X4_250")
    ap.add_argument("--height", type=float, default=FLIGHT_HEIGHT, help="высота полёта, м")
    ap.add_argument("--simulator", action="store_true", help="Pioneer(simulator=True)")
    ap.add_argument("--debug-dir", default=None, help="папка для сохранения размеченных кадров")
    ap.add_argument("--pose-only", action="store_true", help="не летать, только печатать позу")
    ap.add_argument("--find-dict", action="store_true", help="подобрать словарь по кадру с дрона")
    ap.add_argument("--image", default=None, help="оценить позу по файлу изображения и выйти")
    args = ap.parse_args()

    FLIGHT_HEIGHT = args.height

    markers = load_map(args.map)
    print("[map] загружено %d маркеров из %s" % (len(markers), args.map))
    cam_model = CameraModel(args.calib)
    detect = make_detector(args.dict)

    # --- оценка по файлу (без дрона) ---
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit("Не удалось прочитать %s" % args.image)
        if args.find_dict:
            find_dictionary(img)
            return
        p = estimate_pose(img, detect, markers, cam_model, args.debug_dir and os.path.join(args.debug_dir, "img.jpg"))
        print("поза:", None if p is None else p.xyz, "маркеров:", None if p is None else p.n)
        return

    from pioneer_sdk import Pioneer, Camera   # импорт здесь, чтобы файл работал без SDK

    camera = Camera()

    if args.find_dict:
        f = grab_one_frame(camera)
        camera.disconnect()
        if f is None:
            sys.exit("Нет кадра с камеры")
        find_dictionary(f)
        return

    poses = PoseSource(camera, detect, markers, cam_model, args.debug_dir)

    if args.pose_only:
        print("Режим --pose-only, Ctrl+C для выхода")
        try:
            while True:
                p = poses.get_pose(timeout=3.0)
                print("нет позы" if p is None else
                      "карта: x=%.3f y=%.3f z=%.3f | маркеров=%d | reproj=%.1fpx" %
                      (p.xyz[0], p.xyz[1], p.xyz[2], p.n, p.reproj))
        except KeyboardInterrupt:
            pass
        finally:
            camera.disconnect()
        return

    drone = Pioneer(simulator=True) if args.simulator else Pioneer()
    time.sleep(3)
    nav = DroneNav(drone, poses, markers)
    try:
        nav.takeoff()
        poses.drain(FRAME_DRAIN_TIME)
        if poses.get_pose() is None:
            print("[init] после взлёта карта не видна. Проверьте словарь (--find-dict), "
                  "высоту, освещённость, расположение дрона над полем.")
            return
        if not nav.calibrate_axes():
            print("[init] калибровка осей не удалась, посадка")
            return
        prompt_loop(nav, markers)
    except KeyboardInterrupt:
        print("\nCtrl+C: посадка")
    finally:
        nav.land()
        time.sleep(3)
        try:
            camera.disconnect()
        except Exception:
            pass
        drone.close_connection()


if __name__ == "__main__":
    main()
