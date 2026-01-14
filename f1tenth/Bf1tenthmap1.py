import os
import cv2
import csv
import yaml
import numpy as np
import sys
import math
import heapq
import time
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.interpolate import UnivariateSpline

# =========================================================
# 1. CLASE LPA*
# =========================================================
class LPAStar:
    def __init__(self, start, goal, width, height, obstacles, visualize=False, ax=None):
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.visualize = visualize 
        self.ax = ax                
        
        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))
        self.U = [] 

        self.rhs[self.start] = 0
        heapq.heappush(self.U, self.calculate_key(self.start))

        self.visited_x = []
        self.visited_y = []

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def calculate_key(self, s):
        g_val = self.g[s]
        rhs_val = self.rhs[s]
        min_val = min(g_val, rhs_val)
        k1 = min_val + self.heuristic(s, self.goal)
        k2 = min_val
        return (k1, k2, s)

    def get_neighbors(self, s):
        neighbors = []
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in moves:
            nx, ny = s[0] + dx, s[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    neighbors.append((nx, ny))
        return neighbors

    def update_vertex(self, u):
        if u != self.start:
            min_rhs = float('inf')
            for s_prime in self.get_neighbors(u):
                step_cost = math.hypot(u[0]-s_prime[0], u[1]-s_prime[1])
                if self.g[s_prime] != float('inf'):
                     min_rhs = min(min_rhs, self.g[s_prime] + step_cost)
            self.rhs[u] = min_rhs

        self.U = [item for item in self.U if item[2] != u]
        heapq.heapify(self.U)

        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, self.calculate_key(u))

    def compute_shortest_path(self):
        max_iterations = 200000 
        iter_count = 0
        
        if self.visualize:
            print("Visualizando expansión...")

        while self.U and iter_count < max_iterations:
            iter_count += 1
            k_top = self.U[0]
            u = k_top[2]
            
            if self.visualize and self.ax:
                self.visited_x.append(u[0])
                self.visited_y.append(u[1])
                
                if iter_count % 500 == 0: 
                    self.ax.plot(self.visited_x, self.visited_y, 's', color='#A9A9A9', markersize=4.5, markeredgewidth=0)
                    self.ax.set_title(f"LPA* Explorando... Nodos: {iter_count}")
                    plt.pause(0.001) 
                    self.visited_x = []
                    self.visited_y = []

            if k_top >= self.calculate_key(self.goal) and self.rhs[self.goal] == self.g[self.goal]:
                break
                
            heapq.heappop(self.U)
            
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def get_path(self):
        if self.g[self.goal] == float('inf'):
            return []
        path = [self.goal]
        curr = self.goal
        while curr != self.start:
            min_g = float('inf')
            next_node = None
            for s_prime in self.get_neighbors(curr):
                step_cost = math.hypot(curr[0]-s_prime[0], curr[1]-s_prime[1])
                val = self.g[s_prime] + step_cost
                if val < min_g:
                    min_g = val
                    next_node = s_prime
            if next_node:
                curr = next_node
                path.append(curr)
            else:
                break
        return path[::-1]


# =========================================================
# 2. LOGICA DE CENTRADO Y SUAVIZADO PERFECCIONADA
# =========================================================
def center_and_smooth(points_world, img_path_full, resolution, origin, smoothing_factor=0.5):
    if not points_world:
        return []

    # --- PASO 0: PRE-FILTRADO AJUSTADO ---
    # Bajamos a 0.8m para capturar mejor la geometría de las curvas,
    # el smoothing_factor alto se encargará de borrar el ruido.
    min_dist_filter = 0.8               # ========================================================= 1.5     0.8
    filtered_points = [points_world[0]]
    
    for i in range(1, len(points_world)):
        p_prev = filtered_points[-1]
        p_curr = points_world[i]
        dist = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        
        if dist >= min_dist_filter:
            filtered_points.append(p_curr)
    
    if filtered_points[-1] != points_world[-1]:
        filtered_points.append(points_world[-1])

    # --- PASO 1: MAPA DE DISTANCIAS ---
    img = cv2.imread(str(img_path_full), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: No se pudo cargar imagen para suavizado.")
        return points_world
    
    h, w = img.shape
    _, binary_map = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
    
    centered_points = []
    search_range = 10 

    # --- PASO 2: CENTRADO ---
    for xw, yw in filtered_points:
        px = int((xw - origin[0]) / resolution)
        py = int(h - (yw - origin[1]) / resolution) 
        
        if 0 <= py < h and 0 <= px < w:
            y_min = max(0, py - search_range)
            y_max = min(h, py + search_range)
            x_min = max(0, px - search_range)
            x_max = min(w, px + search_range)
            
            roi = dist_map[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                _, _, _, max_loc = cv2.minMaxLoc(roi)
                new_px = x_min + max_loc[0]
                new_py = y_min + max_loc[1]
                new_xw = (new_px * resolution) + origin[0]
                new_yw = (h - new_py) * resolution + origin[1]
                centered_points.append([new_xw, new_yw])
            else:
                centered_points.append([xw, yw])
        else:
            centered_points.append([xw, yw])

    # --- PASO 3: SPLINES ---
    return smooth_trajectory_spline(centered_points, smoothing_factor)

def smooth_trajectory_spline(points, smoothing_factor=0.5, degree=3, points_per_meter=10):
    if len(points) < 4: return points
    pts = np.array(points)
    
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    distances = np.sqrt(dx**2 + dy**2)
    t = np.concatenate(([0], np.cumsum(distances)))
    
    try:
        # Generar Splines
        spl_x = UnivariateSpline(t, pts[:, 0], k=degree, s=smoothing_factor)
        spl_y = UnivariateSpline(t, pts[:, 1], k=degree, s=smoothing_factor)
        
        total_dist = t[-1]
        t_new = np.linspace(0, total_dist, int(total_dist * points_per_meter))
        
        return np.vstack((spl_x(t_new), spl_y(t_new))).T.tolist()
    except Exception as e:
        print(f"Error en Spline: {e}")
        return points


# ================= FUNCIONES BASE =================
def load_map_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_map_grid(img_path, downsample_factor=1):
    map_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < 230] = 1 

    if downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)

    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    map_bin = cv2.resize(map_bin, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_NEAREST)
    map_bin = (map_bin > 0.5).astype(np.uint8)
    return map_bin

def grid_from_map(map_bin):
    h, w = map_bin.shape
    obstacles = { (x, h - 1 - y) for y in range(h) for x in range(w) if map_bin[y, x] == 1 }
    return obstacles

def world_to_map(x_world, y_world, resolution, origin):
    return (int((x_world - origin[0]) / resolution), int((y_world - origin[1]) / resolution))

def map_to_world(x_map, y_map, resolution, origin):
    return (x_map * resolution + origin[0], y_map * resolution + origin[1])

def resample_path_fixed(path_world, step_meters):
    if not path_world: return []
    resampled = [path_world[0]]
    acc_dist = 0.0
    for i in range(1, len(path_world)):
        p0, p1 = path_world[i-1], path_world[i]
        dist = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        while acc_dist + dist >= step_meters:
            ratio = (step_meters - acc_dist) / dist
            nx = p0[0] + ratio * (p1[0] - p0[0])
            ny = p0[1] + ratio * (p1[1] - p0[1])
            resampled.append((nx, ny))
            p0 = (nx, ny)
            dist -= (step_meters - acc_dist)
            acc_dist = 0.0
        acc_dist += dist
    if resampled[-1] != path_world[-1]: resampled.append(path_world[-1])
    return resampled

def save_csv(path, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for p in path: writer.writerow(p)

# ================= MAIN =================
if __name__ == "__main__":
    
    HERE = Path(__file__).resolve().parent
    yaml_path = HERE.parent / "Mapas-F1Tenth" / "Oschersleben_map.yaml"
    
    REQUIRED_SPACING = 1.0
    
    # ===============================================
    # ⚠️ FACTOR DE SUAVIZADO AUMENTADO A 15
    # Esto elimina cualquier imperfección restante
    # ===============================================
    SMOOTHING_FACTOR = 6            # =============================================== 2 
    
    map_cfg = load_map_config(yaml_path)
    resolution_orig = map_cfg['resolution']
    origin = map_cfg['origin']
    img_path = (yaml_path.parent / map_cfg['image']).resolve()

    downsample = 4
    resolution_grid = resolution_orig * downsample
    map_bin = load_map_grid(img_path, downsample)
    obstacles = grid_from_map(map_bin)
    h_grid, w_grid = map_bin.shape

    x_start, y_start = -21.0, -4.0
    x_goal,  y_goal  = -19.0, -4.7
    start_node = world_to_map(x_start, y_start, resolution_grid, origin)
    goal_node  = world_to_map(x_goal,  y_goal,  resolution_grid, origin)

    print("Calculando ruta LPA*...")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if obstacles:
        ox = [x for x,y in obstacles]
        oy = [y for x,y in obstacles]
        ax.plot(ox, oy, 'ks', markersize=3.5)
    
    planner = LPAStar(start_node, goal_node, w_grid, h_grid, obstacles, visualize=True, ax=ax)
    start_time = time.time()
    planner.compute_shortest_path()
    path_grid = planner.get_path()
    
    if not path_grid:
        print("Fallo al encontrar ruta.")
    else:
        raw_path_world = [map_to_world(p[0], p[1], resolution_grid, origin) for p in path_grid]
        
        print("Aplicando Centrado y Suavizado PRO...")
        smooth_path_world = center_and_smooth(
            raw_path_world, 
            img_path, 
            resolution_orig, 
            origin, 
            smoothing_factor=SMOOTHING_FACTOR
        )
        
        final_waypoints = resample_path_fixed(smooth_path_world, REQUIRED_SPACING)
        end_time = time.time()

        # DIBUJAR
        sx = []
        sy = []
        for p in smooth_path_world:
            pm = world_to_map(p[0], p[1], resolution_grid, origin)
            sx.append(pm[0])
            sy.append(pm[1])
            
        ax.plot(sx, sy, 'c-', linewidth=2.5, label='Suavizado Spline Pro')
        
        ax.plot(start_node[0], start_node[1], 'go', markersize=8)
        ax.plot(goal_node[0], goal_node[1], 'bo', markersize=8)
        ax.legend()
        
        plt.ioff()
        
        csv_name = f"lpastar_centered_{str(REQUIRED_SPACING).replace('.', '')}m.csv"
        save_csv(final_waypoints, csv_name)
        
        print("-" * 30)
        print(f"✅ Proceso Completado")
        print(f"🔹 Waypoints: {len(final_waypoints)}")
        print(f"⏱️ Tiempo: {end_time - start_time:.4f}s")
        print(f"📂 Archivo: {csv_name}")
        print("-" * 30)
        
    plt.show()
