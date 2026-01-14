import os
import cv2
import csv
import yaml
import numpy as np
import sys
import math
import heapq
import time  # Import necesario para el tiempo
import matplotlib.pyplot as plt 
from pathlib import Path
from collections import defaultdict

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

        # Listas para guardar puntos explorados para la animación
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
            print("Visualizando expansión (Celdas exploradas)...")

        while self.U and iter_count < max_iterations:
            iter_count += 1
            k_top = self.U[0]
            u = k_top[2]
            
            # --- ANIMACIÓN DE CELDAS EXPLORADAS ---
            if self.visualize and self.ax:
                self.visited_x.append(u[0])
                self.visited_y.append(u[1])
                
                # Actualizar cada 200 iteraciones
                if iter_count % 200 == 0: 
                    # GRIS SÓLIDO (Markersize 4.5 para rellenar huecos)
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

    if downsample_factor > 12:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
    elif downsample_factor >= 4:
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


# ================= OBSTÁCULOS =================
def grid_from_map(map_bin):
    h, w = map_bin.shape
    obstacles = {
        (x, h - 1 - y)
        for y in range(h)
        for x in range(w)
        if map_bin[y, x] == 1
    }
    return obstacles


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


# ================= REMUESTREO 0.5 m =================
def resample_path_05m(path_map, resolution, origin, image_height):
    if len(path_map) < 2:
        return path_map

    path_world = [
        map_to_world(x, y, resolution, origin, image_height)
        for x, y in path_map
    ]

    resampled = [path_world[0]]
    acc_dist = 0.0
    step = 0.5 

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

    downsample_factor = 4

    x_start, y_start = -21.0, -4.0
    x_goal,  y_goal  = -19.0, -4.7

    map_bin, resolution, origin = load_map(yaml_path, downsample_factor)
    obstacles = grid_from_map(map_bin)
    h, w = map_bin.shape

    start = world_to_map(x_start, y_start, resolution, origin)
    goal  = world_to_map(x_goal,  y_goal,  resolution, origin)

    print(f"Start (map): {start}, Goal (map): {goal}")

    # =======================================================
    # VISUALIZACIÓN INICIAL
    # =======================================================
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. OBSTÁCULOS (NEGRO SÓLIDO)
    # Aumentamos markersize a 5 para unir los puntos
    if obstacles:
        obs_x = [o[0] for o in obstacles]
        obs_y = [o[1] for o in obstacles]
        ax.plot(obs_x, obs_y, 'ks', markersize=5, label='Obstáculos') 
    
    # 2. INICIO Y META
    ax.plot(start[0], start[1], 'go', markersize=8, label='Inicio')
    ax.plot(goal[0], goal[1], 'bo', markersize=8, label='Meta') 

    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_title("Planificación LPA* (F1Tenth)")
    
    plt.pause(0.5)

    # =======================================================
    # EJECUCIÓN DEL ALGORITMO CON ANIMACIÓN
    # =======================================================
    planner = LPAStar(start, goal, w, h, obstacles, visualize=True, ax=ax)
    
    # ⏱️ INICIO DEL TIEMPO
    start_time = time.time()
    
    planner.compute_shortest_path()
    path_map = planner.get_path()
    
    # ⏱️ FIN DEL TIEMPO
    end_time = time.time()

    if not path_map:
        print("No se encontró camino.")
    else:
        # Dibujar ruta final
        px, py = zip(*path_map)
        ax.plot(px, py, 'r-', linewidth=3, label='Ruta Final')
        
        # Volver a pintar Inicio/Fin encima
        ax.plot(start[0], start[1], 'go', markersize=8)
        ax.plot(goal[0], goal[1], 'bo', markersize=8)

        plt.ioff()
        print("Ruta encontrada.")

        # Guardar CSV y Resamplear
        path_05m = resample_path_05m(
            path_map,
            resolution,
            origin,
            map_bin.shape[0]
        )
        save_path_world_csv(path_05m, "Alpastar_05m.csv")
        print(f"Ruta LPA* guardada → Alpastar_05m.csv")
        
        # =======================================================
        # REPORTE FINAL (SIN DISTANCIA)
        # =======================================================
        REQUIRED_SPACING = 0.5 
        final_waypoints = path_05m

        print("-" * 30)
        # Solo mostramos Waypoints y Tiempo
        print(f"Distancia_Waypoints: {len(final_waypoints)} (~{REQUIRED_SPACING}m)")
        print(f"Tiempo: {end_time - start_time:.4f} s")
        print("-" * 30)
    
    plt.show()
