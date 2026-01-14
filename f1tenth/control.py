import os, csv, yaml, cv2, numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Grid, SearchFactory, ControlFactory
from pathlib import Path

def load_map(yaml_path, downsample_factor=1):
    yaml_path = Path(yaml_path)  # asegurar Path
    with yaml_path.open('r') as f:
        map_config = yaml.safe_load(f)


    img_path = Path(map_config['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()
    map_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    # Binarizar: 1 = ocupado, 0 = libre
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1

    # Engrosar obstáculos según el factor
    if downsample_factor > 12:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
    elif downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)
    # para 1-3 no se dilata

    # Downsampling con interpolación adecuada
    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    map_bin = cv2.resize(map_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Re-binarizar según nivel
    if downsample_factor > 12:
        map_bin = (map_bin > 0.10).astype(np.uint8)
    elif downsample_factor >= 4:
        map_bin = (map_bin > 0.25).astype(np.uint8)
    else:
        map_bin = (map_bin >= 0.5).astype(np.uint8)

    # Ajustar resolución
    resolution *= downsample_factor

    return map_bin, resolution, origin


def grid_from_map(map_bin):
    h, w = map_bin.shape
    env = Grid(w, h)
    obstacles = {(x, h - 1 - y) for y in range(h) for x in range(w) if map_bin[y, x] == 1}
    env.update(obstacles)
    return env


def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)


def map_to_world(x_map, y_map, resolution, origin, image_height):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)

def is_free_cell(cell, map_bin):
    h, w = map_bin.shape
    i, j = cell
    if not (0 <= i < w and 0 <= j < h): 
        return False
    # map_bin está en coords de imagen => invertir Y para indexar
    return map_bin[h - 1 - j, i] == 0

# --- tu caso ---
if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    yaml_path = HERE.parent / "Mapas-F1Tenth" / "example_map.yaml"
    factor = 16

    # 1) mapa -> grid
    map_bin, resolution, origin = load_map(yaml_path, downsample_factor=factor)
    env = grid_from_map(map_bin)

    # 2) start/goal (en metros) -> celdas
    x_start, y_start = 0.0, 1.0
    x_goal,  y_goal  = -0.5, -1.5
    start = world_to_map(x_start, y_start, resolution, origin)
    goal  = world_to_map(x_goal,  y_goal,  resolution, origin)

    print(f"Start (map): {start}, Goal (map): {goal}")
    if not is_free_cell(start, map_bin) or not is_free_cell(goal, map_bin):
        raise ValueError("Start o Goal caen en obstáculo tras el downsampling. Ajusta posiciones/umbral.")

    # 3) ruta global con A* (start->goal)
    g_planner = SearchFactory()("a_star", start=start, goal=goal, env=env)
    g_cost, g_path, _ = g_planner.plan()
    if not g_path:
        raise RuntimeError("A* no encontró ruta. Revisa umbral/posición/factor.")

    # 4) crear PID e INYECTAR la ruta (clave)
    start_2 = (start[0], start[1], 1.57)
    goal_2  = (goal[0],  goal[1],  1.57)
    pid = ControlFactory()("pid",
                start=start_2,
                goal=goal_2,
                env=env,
                LOG_FILE="velocities.csv",
                TIME_STEP=0.1,
                path=g_path[::-1],          # A* da goal->start, se invierte
                resolution=resolution,    
                origin=origin              
            )

    # El PID espera self.path en coords de grid de start->goal
    pid.path = g_path[::-1]  # A* suele devolver goal->start

    # Ajustes útiles para factor grande (~1 m/celda)
    pid.params["MAX_ITERATION"] = max(500000, pid.params.get("MAX_ITERATION", 2000))
    pid.params["MIN_LOOKAHEAD_DIST"] = 5.0
    pid.params["MAX_LOOKAHEAD_DIST"] = 5.0
    pid.params["MAX_V"] = 2.0
    pid.params["MAX_W"] = 1.0

    # 5) correr animación
    pid.run()
# if __name__ == "__main__":
    yaml_path = "example_map.yaml"
    factor = 8

    # 1) mapa -> grid
    map_bin, resolution, origin = load_map(yaml_path, downsample_factor=factor)
    env = grid_from_map(map_bin)

    # 2) start/goal en mundo (m) -> celdas (ix,iy)
    x_start, y_start = 0.0,  1.0
    x_goal,  y_goal  = 0.0, -1.5
    start = world_to_map(x_start, y_start, resolution, origin)
    goal  = world_to_map(x_goal,  y_goal,  resolution, origin)

    print(f"Start (map): {start}, Goal (map): {goal}")
    if not is_free_cell(start, map_bin) or not is_free_cell(goal, map_bin):
        raise ValueError("Start o Goal caen en obstáculo tras el downsampling. Ajusta posiciones/umbral.")

    # 3) ruta global con A* (start->goal)
    g_planner = SearchFactory()("a_star", start=start, goal=goal, env=env)
    g_cost, g_path, _ = g_planner.plan()
    if not g_path:
        raise RuntimeError("A* no encontró ruta. Revisa umbral/posición/factor.")

    # 4) crear LQR e INYECTAR la ruta
    start_2 = (start[0], start[1], 0.0)  # (ix,iy,theta)
    goal_2  = (goal[0],  goal[1],  0.0)
    lqr = ControlFactory()("lqr", start=start_2, goal=goal_2, env=env,
              path=g_path[::-1],              # A* -> start->goal
              min_lookahead_dist=1.0)         # según tu grid

    # Ajustes útiles (igual que hiciste con PID)
    lqr.params["MAX_ITERATION"] = max(10000, lqr.params.get("MAX_ITERATION", 2000))
    lqr.params["MIN_LOOKAHEAD_DIST"] = max(1.0, resolution)  # ~1 celda
    lqr.params["MAX_V"] = 5.0
    lqr.params["MAX_W"] = 5.5

    # (Opcional) tunning LQR:
    # lqr.Q = np.diag([2.0, 2.0, 0.5])
    # lqr.R = np.diag([0.5, 0.8])
    # lqr.lqr_iteration = 200
    # lqr.eps_iter = 1e-3

    # 5) correr
    lqr.run()