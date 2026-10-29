#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_camera.py - калибровка камеры Pioneer Mini по шахматной доске (аналог ROS cameracalibrator).

Источники кадров:
  по умолчанию        - живая камера дрона через pioneer_sdk.Camera (Wi-Fi, с задержкой)
  --images папка      - офлайн: калибровка по уже сохранённым картинкам

Управление в окне:
  SPACE / s  сохранить кадр (только если доска найдена)
  a          вкл/выкл авто-съёмку (--auto делает то же с самого начала)
  c          выполнить калибровку и сохранить результат
  u          показать/скрыть исправленную (undistort) картинку после калибровки
  ESC / q    выход

Доска: --cols и --rows это число ВНУТРЕННИХ углов (для make_chessboard.py по умолчанию 9x6, клетка 25 мм).
Размер клетки (--square, мм) должен совпадать с реально измеренным на печати.

Результат: camera_calib.npz (K, dist, image_size, rms) и camera_calib.yaml (в стиле ROS camera_info).
Для дронов с очень широким объективом попробуйте --rational (8 коэффициентов).
"""

import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

# ROS cameracalibrator: диапазоны покрытия параметров (x, y, size, skew)
PARAM_RANGES = (0.7, 0.7, 0.4, 0.5)


def find_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def board_params(corners, pattern, size):
    """Те же 4 параметра, что считает ROS: x, y, size, skew (нормированные)."""
    w, h = size
    c = corners.reshape(-1, 2)
    cols, rows = pattern
    up_left = c[0]
    up_right = c[cols - 1]
    down_right = c[-1]
    down_left = c[cols * (rows - 1)]
    a, b = up_right - up_left, down_left - up_left
    area = 0.5 * abs(np.cross(down_right - up_left, down_left - up_right))
    border = math.sqrt(area)
    p_x = min(1.0, max(0.0, (np.mean(c[:, 0]) - border / 2) / (w - border)))
    p_y = min(1.0, max(0.0, (np.mean(c[:, 1]) - border / 2) / (h - border)))
    p_size = math.sqrt(area / (w * h))
    cosang = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    skew = min(1.0, 2.0 * abs(math.pi / 2 - math.acos(max(-1.0, min(1.0, cosang)))))
    return np.array([p_x, p_y, p_size, skew])


def progress(params_list):
    if not params_list:
        return [0, 0, 0, 0]
    P = np.array(params_list)
    return [min((P[:, i].max() - P[:, i].min()) / PARAM_RANGES[i], 1.0) for i in range(4)]


def sufficiently_new(p, params_list):
    """ROS-подобная проверка: кадр должен заметно отличаться от уже взятых."""
    if not params_list:
        return True
    d = np.min(np.linalg.norm((np.array(params_list) - p) / np.array(PARAM_RANGES), axis=1))
    return d > 0.12


def calibrate(objpoints, imgpoints, size, rational, prune=True):
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None, flags=flags)
    errs = []
    for o, i, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    if prune and len(errs) > 10:
        thr = max(2.0 * float(np.median(errs)), 0.8)
        keep = [k for k, e in enumerate(errs) if e <= thr]
        if 8 <= len(keep) < len(errs):
            print("Исключаю %d плохих кадров (ошибка > %.2f px) и пересчитываю" % (len(errs) - len(keep), thr))
            return calibrate([objpoints[k] for k in keep], [imgpoints[k] for k in keep], size, rational, prune=False)
    return rms, K, dist, errs


def save_result(path, rms, K, dist, size, pattern, square_mm, n):
    base = os.path.splitext(path)[0]
    np.savez(path, K=K, dist=dist.ravel(), image_size=np.array(size), rms=rms)
    fs = cv2.FileStorage(base + ".yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(size[0]))
    fs.write("image_height", int(size[1]))
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist.reshape(1, -1))
    fs.write("rms_reprojection_error", float(rms))
    fs.write("board_inner_corners", "%dx%d" % pattern)
    fs.write("square_size_mm", float(square_mm))
    fs.write("num_images", int(n))
    fs.release()
    print("Сохранено: %s и %s.yaml" % (path, base))


def run_calibration(objp, samples, size, args, pattern):
    if len(samples) < 8:
        print("Мало кадров (%d). Нужно минимум 8, лучше 20+." % len(samples))
        return None
    objpoints = [objp] * len(samples)
    rms, K, dist, errs = calibrate(objpoints, samples, size, args.rational)
    print("\n=== РЕЗУЛЬТАТ ===")
    print("RMS ошибка репроекции: %.3f px  (хорошо: < 0.5, допустимо: < 1.0)" % rms)
    print("Размер изображения: %dx%d" % size)
    print("K =\n%s" % np.array2string(K, precision=2, suppress_small=True))
    print("dist =", np.array2string(dist.ravel(), precision=5, suppress_small=True))
    hfov = 2 * math.degrees(math.atan(size[0] / (2 * K[0, 0])))
    vfov = 2 * math.degrees(math.atan(size[1] / (2 * K[1, 1])))
    print("Оценка угла обзора: %.0f° x %.0f°" % (hfov, vfov))
    if rms > 1.0:
        print("ВНИМАНИЕ: ошибка большая. Проверьте плоскость доски, размер клетки, добавьте кадры "
              "с наклонами и по краям, попробуйте --rational.")
    save_result(args.out, rms, K, dist, size, pattern, args.square, len(samples))
    return K, dist


def offline(args, pattern, objp):
    files = sorted(sum([glob.glob(os.path.join(args.images, e)) for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    if not files:
        sys.exit("В %s нет изображений" % args.images)
    samples, size = [], None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (g.shape[1], g.shape[0])
        c = find_corners(g, pattern)
        print("%-40s %s" % (os.path.basename(f), "OK" if c is not None else "доска не найдена"))
        if c is not None:
            samples.append(c)
    run_calibration(objp, samples, size, args, pattern)


def online(args, pattern, objp):
    from pioneer_sdk import Camera
    camera = Camera()
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
    samples, params, saved = [], [], 0
    auto, K, dist, show_undist = args.auto, None, None, False
    last_frame, last_auto = None, 0.0
    win = "Pioneer calibration"
    print("Держите доску перед камерой дрона (дрон на столе, питание включено).")
    print("Покрывайте: центр, углы кадра, ближе/дальше, наклоны. Нужно 20+ кадров.")
    try:
        while True:
            try:
                frame = camera.get_cv_frame()
            except Exception as e:
                print("ошибка кадра:", e)
                frame = None
            if frame is None:
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            if last_frame is not None and frame.shape == last_frame.shape and np.array_equal(frame, last_frame):
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            last_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            corners = find_corners(gray, pattern)
            vis = frame.copy()
            if show_undist and K is not None:
                vis = cv2.undistort(vis, K, dist)
            if corners is not None:
                cv2.drawChessboardCorners(vis, pattern, corners, True)
            prog = progress(params)
            for i, (name, v) in enumerate(zip(("X", "Y", "Size", "Skew"), prog)):
                cv2.putText(vis, name, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.rectangle(vis, (70, 12 + 22 * i), (70 + int(120 * v), 28 + 22 * i), (0, 200, 0), -1)
                cv2.rectangle(vis, (70, 12 + 22 * i), (190, 28 + 22 * i), (255, 255, 255), 1)
            cv2.putText(vis, "samples: %d  auto:%s  [SPACE]=save [c]=calibrate [q]=quit" %
                        (len(samples), "ON" if auto else "off"), (10, size[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            take = False
            if corners is not None:
                p = board_params(corners, pattern, size)
                if key in (32, ord("s")):
                    take = True
                elif auto and time.time() - last_auto > 1.5 and sufficiently_new(p, params):
                    take = True
                if take:
                    samples.append(corners)
                    params.append(p)
                    last_auto = time.time()
                    if args.save_images:
                        cv2.imwrite(os.path.join(args.save_images, "calib_%03d.png" % saved), frame)
                        saved += 1
                    print("кадр %d сохранён (покрытие X/Y/Size/Skew: %s)" %
                          (len(samples), " ".join("%.0f%%" % (100 * v) for v in progress(params))))
            elif key in (32, ord("s")):
                print("доска не найдена в кадре")

            if key == ord("a"):
                auto = not auto
            elif key == ord("u"):
                show_undist = not show_undist
            elif key == ord("c"):
                res = run_calibration(objp, samples, size, args, pattern)
                if res:
                    K, dist = res
                    show_undist = True
            elif key in (27, ord("q")):
                break
    finally:
        try:
            camera.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Калибровка камеры Pioneer по шахматной доске")
    ap.add_argument("--cols", type=int, default=9, help="внутренних углов по горизонтали")
    ap.add_argument("--rows", type=int, default=6, help="внутренних углов по вертикали")
    ap.add_argument("--square", type=float, default=25.0, help="сторона клетки, мм (измерьте линейкой на печати!)")
    ap.add_argument("--images", default=None, help="папка с картинками (офлайн-режим)")
    ap.add_argument("--save-images", default="calib_images", help="куда сохранять снятые кадры ('' - не сохранять)")
    ap.add_argument("--out", default="camera_calib.npz", help="файл результата")
    ap.add_argument("--auto", action="store_true", help="автоматическая съёмка при смене положения доски")
    ap.add_argument("--rational", action="store_true", help="модель с 8 коэффициентами (широкоугольные объективы)")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * (args.square / 1000.0)

    if args.images:
        offline(args, pattern, objp)
    else:
        online(args, pattern, objp)


if __name__ == "__main__":
    main()
