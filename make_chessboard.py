#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chessboard.py - шахматная доска для калибровки в виде PDF строго в масштабе (A4, альбомная).

По умолчанию: 10x7 клеток по 25 мм = 9x6 внутренних углов (как pattern.png из OpenCV),
размер доски 250x175 мм - помещается на A4.

Печатать в масштабе 100% (без "по размеру страницы"), затем измерить линейкой реальный размер
клетки и передать его в calibrate_camera.py через --square.
Клеить на плоскую жёсткую основу (пенокартон, стекло), бумага не должна коробиться.

Пример:  python make_chessboard.py --square 25 --out chessboard_A4.pdf
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

DPI = 300
A4_MM = (297.0, 210.0)  # альбомная ориентация


def mm(v):
    return int(round(v / 25.4 * DPI))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="внутренних углов по горизонтали")
    ap.add_argument("--rows", type=int, default=6, help="внутренних углов по вертикали")
    ap.add_argument("--square", type=float, default=25.0, help="клетка, мм")
    ap.add_argument("--out", default="chessboard_A4.pdf")
    args = ap.parse_args()

    nx, ny = args.cols + 1, args.rows + 1
    bw, bh = nx * args.square, ny * args.square
    if bw > A4_MM[0] - 10 or bh > A4_MM[1] - 20:
        raise SystemExit("Доска %.0fx%.0f мм не помещается на A4 альбомной" % (bw, bh))

    W, H = mm(A4_MM[0]), mm(A4_MM[1])
    page = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(page)
    x0 = (W - mm(bw)) // 2
    y0 = (H - mm(bh)) // 2 - mm(4)
    s = args.square
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                d.rectangle([x0 + mm(i * s), y0 + mm(j * s),
                             x0 + mm((i + 1) * s) - 1, y0 + mm((j + 1) * s) - 1], fill=0)
    label = "Chessboard %dx%d inner corners, square %.1f mm, print at 100%% scale (A4 landscape)" % (
        args.cols, args.rows, s)
    d.text((x0, y0 + mm(bh) + mm(5)), label, fill=0)
    page.save(args.out, "PDF", resolution=float(DPI))
    print("Сохранено %s: доска %.0fx%.0f мм, %dx%d внутренних углов" % (args.out, bw, bh, args.cols, args.rows))


if __name__ == "__main__":
    main()
