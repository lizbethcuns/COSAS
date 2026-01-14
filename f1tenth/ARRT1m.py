import os
import cv2
import csv
import yaml
import numpy as np
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Map, SearchFactory

# ===================== MAP LOADING =====================
def load_map(yaml_path, downsample_factor=1):
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as f:
        map_config = yaml.safe_load(f)

    img_path = Path(map_config['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()

    map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    # ---------- BINARIZACIÓN ----------
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1

    # ---------- ENGROSAR OBSTÁCULOS ----------
    map_bin = cv2.dilate(
        map_bin,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # ---------- DOWNSAMPLE ----------
    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    map_bin = cv2.resize(
        map_bin,
        (w // downsample_factor, h // downsample_factor),
        interpolation=cv2.INTER_AREA
    )

    # ---------- RE-BINARIZACIÓN (CLAVE) ----------
    map_bin = (map_bin > 0.35).astype(np.uint8)

    resolution *= downsample_factor
    return map_bin, resolution, origin

# ===================== MAP → RRT ENV =====================
def map_from_binary(map_bin):
    h, w = map_bin.shape
    env = Map(w, h)
    obs_rect = []

    for y in range(h):
        for x in range(w):
            if map_bin[y, x] == 1:
                obs_rect.append([x, h - 1 - y, 1, 1])

    env.update(obs_rect=obs_rect)
    return env

# ===================== COORDENADAS =====================
def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return x_map, y_map

def map_to_world(x_map, y_map, resolution, origin):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return x_world, y_world

# ===================== WAYPOINTS 1.0 m =====================
def resample_path_rrt(path, resolution, origin, step=1.0):
    if not path:
        return []

    resampled = [path[0]]
    last_xw, last_yw = map_to_world(*path[0], resolution, origin)

    for x_map, y_map in path[1:]:
        xw, yw = map_to_world(x_map, y_map, resolution, origin)
        dist = np.hypot(xw - last_xw, yw - last_yw)

        if dist >= step:
            resampled.append((x_map, y_map))
            last_xw, last_yw = xw, yw

    if path[-1] not in resampled:
        resampled.append(path[-1])

    return resampled

# ===================== SAVE CSV =====================
def save_path_as_csv(path, filename, resolution, origin):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x_map, y_map in path:
            x, y = map_to_world(x_map, y_map, resolution, origin)
            writer.writerow([x, y])

# ===================== MAIN =====================
if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    yaml_path = HERE.parent / "Mapas-F1Tenth" / "Oschersleben_map.yaml"

    downsample_factor = 6

    x_start, y_start = -21.0, -4.0
    x_goal,  y_goal  = -19.0, -4.7

    map_bin, resolution, origin = load_map(yaml_path, downsample_factor)
    env = map_from_binary(map_bin)

    start = world_to_map(x_start, y_start, resolution, origin)
    goal  = world_to_map(x_goal, y_goal, resolution, origin)

    print(f"Start (map): {start}, Goal (map): {goal}")
    print(f"Resolución: {resolution:.3f} m/celda")

    # ===================== RRT =====================
    planner = SearchFactory()(
        "rrt",
        start=start,
        goal=goal,
        env=env,
        max_dist=4,          # <<< AJUSTE CLAVE
        sample_num=150000
    )

    planner.run()
    cost, path, _ = planner.plan()

    if path:
        path_05 = resample_path_rrt(path, resolution, origin, step=1.0)
        save_path_as_csv(path_05, "Arrt_1m.csv", resolution, origin)
        print(f"Ruta RRT 1.0 m guardada ({len(path_05)} waypoints)")
    else:
        print("RRT no encontró camino")

