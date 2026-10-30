"""Генератор карты ArUco

Использование:
  python genmap.py <length> <x> <y> <dist_x> <dist_y> [<first>] [<x0>] [<y0>] [--top-left | --bottom-left] [-o <filename>]

Аргументы:
  <length>       Размер стороны маркера (метры)
  <x>            Количество маркеров по X
  <y>            Количество маркеров по Y
  <dist_x>       Расстояние между центрами маркеров по X
  <dist_y>       Расстояние между центрами маркеров по Y
  <first>        ID первого маркера [по дефолту: 0]
  <x0>           X координата первого маркера [по дефолту: 0] (экспериментально)
  <y0>           Y координата первого маркера [по дефолту: 0] (экспериментально)

Пример:
  python genmap.py 0.33 2 4 1 1 0 -o test_map.txt

Для карты соревнований:
    python genmap.py 0.200 9 19 0.4 0.4 0 -o fsp.txt
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate markers map')
    parser.add_argument('length', type=float, help='Marker side length')
    parser.add_argument('x', type=int, help='Marker count along X axis')
    parser.add_argument('y', type=int, help='Marker count along Y axis')
    parser.add_argument('dist_x', type=float, help='Distance between markers along X axis')
    parser.add_argument('dist_y', type=float, help='Distance between markers along Y axis')
    parser.add_argument('first', type=int, nargs='?', default=0, help='First marker ID [default: 0]')
    parser.add_argument('x0', type=float, nargs='?', default=0, help='X coordinate for the first marker [default: 0]')
    parser.add_argument('y0', type=float, nargs='?', default=0, help='Y coordinate for the first marker [default: 0]')
    parser.add_argument('-o', '--output', help='Output map file name')
    parser.add_argument('--bottom-left', action='store_true', help='First marker is on bottom-left (default: top-left)')

    args = parser.parse_args()

    if args.output is None:
        output = sys.stdout
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, args.output)
        output = open(output_path, 'w')

    max_y = args.y0 + (args.y - 1) * args.dist_y

    output.write('# id\tlength\tx\ty\tz\trot_z\trot_y\trot_x\n')
    for y in range(args.y):
        for x in range(args.x):
            pos_x = args.x0 + x * args.dist_x
            pos_y = args.y0 + y * args.dist_y
            if not args.bottom_left:
                pos_y = max_y - pos_y
            output.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                args.first, args.length, pos_x, pos_y, 0, 0, 0, 0))
            args.first += 1

    if args.output is not None:
        output.close()
        print("Map file saved as:", os.path.abspath(output_path))


if __name__ == "__main__":
    main()