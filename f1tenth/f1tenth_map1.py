import os
import cv2
import csv
import yaml
import numpy as np
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_motion_planning.utils import Grid, SearchFactory
from pathlib import Path
import python_motion_planning.curve_generation.cubic_spline as cs


# ================= CARGA DEL MAPA =================
def load_map(yaml_path, downsample_factor=1):
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as f:
        map_config = yaml.safe_load(f)

    img_path = Path(map_config['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()

    map_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1

    if downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)

    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    map_bin = cv2.resize(
        map_bin,
        (w // downsample_factor, h // downsample_factor),
        interpolation=cv2.INTER_AREA
    )

    if downsample_factor > 12:
        map_bin = (map_bin > 0.10).astype(np.uint8)
    elif downsample_factor >= 4:
        map_bin = (map_bin > 0.25).astype(np.uint8)
    else:
        map_bin = (map_bin >= 0.5).astype(np.uint8)

    resolution *= downsample_factor
    return map_bin, resolution, origin


# ================= GRID =================
def grid_from_map(map_bin):
    h, w = map_bin.shape
    env = Grid(w, h)
    obstacles = {
        (x, h - 1 - y)
        for y in range(h)
        for x in range(w)
        if map_bin[y, x] == 1
    }
    env.update(obstacles)
    return env


# ================= CONVERSIONES =================
def world_to_map(x_world, y_world, resolution, origin):
    return (
        int((x_world - origin[0]) / resolution),
        int((y_world - origin[1]) / resolution)
    )


def map_to_world(x_map, y_map, resolution, origin, image_height):
    return (
        x_map * resolution + origin[0],
        y_map * resolution + origin[1]
    )


# ================= REMUESTREO 1.0 m =================
def resample_path_05m(path_map, resolution, origin, image_height):
    if len(path_map) < 2:
        return []

    path_world = [
        map_to_world(x, y, resolution, origin, image_height)
        for x, y in path_map
    ]

    resampled = [path_world[0]]
    acc_dist = 0.0
    step = 1.0

    for i in range(1, len(path_world)):
        x0, y0 = path_world[i - 1]
        x1, y1 = path_world[i]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)

        while acc_dist + seg_len >= step:
            ratio = (step - acc_dist) / seg_len
            xn = x0 + ratio * dx
            yn = y0 + ratio * dy
            resampled.append((xn, yn))
            x0, y0 = xn, yn
            seg_len = math.hypot(x1 - x0, y1 - y0)
            acc_dist = 0.0

        acc_dist += seg_len

    return resampled


# ================= PARTE B: SUAVIZADO =================
def smooth_path_cubic(path_world, ds=0.1):
    if len(path_world) < 3:
        return path_world

    px = [p[0] for p in path_world]
    py = [p[1] for p in path_world]

    rx, ry, ryaw, rk, s = cs.calc_spline_course(px, py, ds=ds)
    return list(zip(rx, ry))


# ================= GUARDAR CSV =================
def save_path_world_csv(path_world, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in path_world:
            writer.writerow([x, y])


# ================= MAIN =================
if __name__ == "__main__":

    HERE = Path(__file__).resolve().parent
    yaml_path = HERE.parent / "Mapas-F1Tenth" / "Oschersleben_map.yaml"

    downsample_factor = 6

    x_start, y_start = -21.0, -4.0
    x_goal,  y_goal  = -19.0, -4.7

    map_bin, resolution, origin = load_map(yaml_path, downsample_factor)
    env = grid_from_map(map_bin)

    start = world_to_map(x_start, y_start, resolution, origin)
    goal  = world_to_map(x_goal,  y_goal,  resolution, origin)

    planner = SearchFactory()("lpa_star", start=start, goal=goal, env=env)
    planner.run()
    cost, path_map, _ = planner.plan()

    # Parte A
    path_05m = resample_path_05m(
        path_map,
        resolution,
        origin,
        map_bin.shape[0]
    )
    save_path_world_csv(path_05m, "Blpastar_1m.csv")

    # Parte B
    path_smooth = smooth_path_cubic(path_05m, ds=0.1)
    save_path_world_csv(path_smooth, "Blpastar_1smooth.csv")

    print("✔ Parte A: Blpastar_1m.csv")
    print("✔ Parte B (suavizado): Blpastar_smooth1.csv")

