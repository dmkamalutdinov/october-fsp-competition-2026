"""Генератор изображеня ArUco карты
Использование:
  python genmapimage.py [-o <filename>]

Пример:
  python genmapimage.py test_map.txt
"""

import cv2
import numpy as np
import argparse
import os


def parse_map_file(filename):
    markers = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                marker_id = int(parts[0])
                length = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                markers.append({
                    'id': marker_id,
                    'length': length,
                    'x': x,
                    'y': y
                })
            except:
                continue
    return markers


def generate_map_image(markers, output_file, image_size=2000, margin=200, draw_axis=True):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

    if draw_axis:
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    else:
        image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    if not markers:
        cv2.imwrite(output_file, image)
        return

    min_x = min(m['x'] for m in markers)
    max_x = max(m['x'] for m in markers)
    min_y = min(m['y'] for m in markers)
    max_y = max(m['y'] for m in markers)

    map_width = max_x - min_x
    map_height = max_y - min_y

    if map_width == 0 or map_height == 0:
        return

    scale_x = (image_size - 2 * margin) / map_width
    scale_y = (image_size - 2 * margin) / map_height
    scale = min(scale_x, scale_y)

    offset_x = (image_size - map_width * scale) / 2
    offset_y = (image_size - map_height * scale) / 2

    def world_to_pixel(world_x, world_y):
        pixel_x = int(offset_x + (world_x - min_x) * scale)
        pixel_y = int(image_size - offset_y - (world_y - min_y) * scale)
        return pixel_x, pixel_y

    for marker in markers:
        marker_id = marker['id']
        length = marker['length']
        center_x, center_y = marker['x'], marker['y']

        marker_size_px = int(length * scale)
        if marker_size_px < 10:
            marker_size_px = 10

        center_px, center_py = world_to_pixel(center_x, center_y)

        x1 = center_px - marker_size_px // 2
        y1 = center_py - marker_size_px // 2
        x2 = x1 + marker_size_px
        y2 = y1 + marker_size_px

        if x1 < 0 or y1 < 0 or x2 > image_size or y2 > image_size:
            continue

        marker_img = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px, marker_img, 1)

        if draw_axis:
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            image[y1:y2, x1:x2] = marker_bgr
        else:
            image[y1:y2, x1:x2] = marker_img

    if draw_axis:
        origin_x, origin_y = world_to_pixel(0, 0)

        end_x, _ = world_to_pixel(1, 0)
        cv2.arrowedLine(image, (origin_x, origin_y), (end_x, origin_y), (0, 0, 255), 2, tipLength=0.1)

        _, end_y = world_to_pixel(0, 1)
        cv2.arrowedLine(image, (origin_x, origin_y), (origin_x, end_y), (0, 255, 0), 2, tipLength=0.1)

    cv2.imwrite(output_file, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output', default='aruco_map.png')
    parser.add_argument('--size', type=int, default=2000)
    parser.add_argument('--margin', type=int, default=200)
    parser.add_argument('--no-axis', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        return

    markers = parse_map_file(args.input_file)

    if not markers:
        return

    generate_map_image(
        markers=markers,
        output_file=args.output,
        image_size=args.size,
        margin=args.margin,
        draw_axis=not args.no_axis
    )


if __name__ == "__main__":
    main()