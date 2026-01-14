import os
import cv2
import csv
import yaml
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Asegúrate de que esta ruta apunte a tu librería de motion planning si es necesaria
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Si SearchFactory da error, asegúrate de tener el archivo utils.py cerca o ajusta el import
try:
    from python_motion_planning.utils import Map, SearchFactory
except ImportError:
    print("Error: No se encontró 'python_motion_planning'. Verifica tus imports.")

# ===================== 1. CARGA DE MAPA (TU CÓDIGO) =====================
def load_map(yaml_path, downsample_factor=1):
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as f:
        map_config = yaml.safe_load(f)

    img_path = Path(map_config['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()

    # Guardamos la ruta de la imagen original para el centrado luego
    raw_img_path = str(img_path)

    map_img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    # Binarización para RRT
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1
    
    # Engrosar obstáculos (Seguridad para RRT)
    map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)

    # Downsample
    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    map_bin = cv2.resize(map_bin, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_AREA)
    map_bin = (map_bin > 0.35).astype(np.uint8)

    resolution *= downsample_factor
    return map_bin, resolution, origin, raw_img_path

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

# ===================== 2. CONVERSIONES DE COORDENADAS =====================
def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return x_map, y_map

def map_to_world(x_map, y_map, resolution, origin):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return x_world, y_world

# ===================== 3. LÓGICA DE CENTRADO Y SUAVIZADO (CÓDIGO AMIGO) =====================

def get_centered_path(points_world, map_img_path, yaml_res, yaml_origin):
    """
    Toma puntos en coordenadas del mundo y los mueve al centro de la pista 
    usando la imagen original (alta resolución) y Distance Transform.
    """
    img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Advertencia: No se pudo cargar imagen para centrado.")
        return points_world

    h, w = img.shape
    
    # Binarizar: Pista (blanco) vs Paredes (negro)
    # Ajusta el umbral si tu mapa es grisáceo (200-250 suele funcionar para mapas limpios)
    _, binary_map = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # Mapa de distancias: el valor de cada pixel es la distancia a la pared más cercana
    dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)

    centered_points = []
    search_range_m = 0.8  # Buscar centro en un radio de 0.8 metros alrededor del punto RRT
    search_px = int(search_range_m / yaml_res)

    for xw, yw in points_world:
        # Convertir mundo -> pixel (imagen original)
        px = int((xw - yaml_origin[0]) / yaml_res)
        # Nota: Las imágenes cargadas con cv2 tienen origen arriba-izquierda.
        # Los mapas de ROS suelen tener origen abajo-izquierda en metadata, 
        # pero al leer la imagen, el eje Y se invierte.
        py = int(h - (yw - yaml_origin[1]) / yaml_res)

        if 0 <= py < h and 0 <= px < w:
            # Definir ventana de búsqueda
            y_min, y_max = max(0, py - search_px), min(h, py + search_px)
            x_min, x_max = max(0, px - search_px), min(w, px + search_px)

            roi = dist_map[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                centered_points.append([xw, yw])
                continue

            # Encontrar el punto con mayor distancia a paredes en esa zona
            _, _, _, max_loc = cv2.minMaxLoc(roi)
            
            # Coordenadas locales -> globales imagen
            new_px = x_min + max_loc[0]
            new_py = y_min + max_loc[1]

            # Convertir pixel -> mundo
            new_xw = (new_px * yaml_res) + yaml_origin[0]
            new_yw = ((h - new_py) * yaml_res) + yaml_origin[1]
            
            centered_points.append([new_xw, new_yw])
        else:
            centered_points.append([xw, yw])
            
    return centered_points

def apply_spline_and_resample(points, separation_distance=1.0, smoothing=10.0):
    """
    Aplica B-Spline y devuelve puntos espaciados exactamente por 'separation_distance'.
    Cumple con el requisito de 0.5m y 1m.
    """
    pts = np.array(points)
    
    # Eliminar duplicados consecutivos para evitar errores de división por cero
    diff = np.diff(pts, axis=0)
    dist_steps = np.sqrt((diff ** 2).sum(axis=1))
    # Nos quedamos con el primer punto y aquellos donde la distancia > 0.01m
    mask = np.insert(dist_steps > 0.01, 0, True) 
    pts = pts[mask]

    if len(pts) < 3:
        return pts # No se puede hacer spline con menos de 3 puntos

    # Parametrización por distancia acumulada
    x = pts[:, 0]
    y = pts[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    t = np.concatenate(([0], np.cumsum(ds)))
    
    total_length = t[-1]
    
    # Generar Splines (k=3 es cúbico)
    # s es el factor de suavizado. Más alto = más suave pero menos fiel a los puntos originales.
    try:
        spl_x = UnivariateSpline(t, x, k=3, s=smoothing)
        spl_y = UnivariateSpline(t, y, k=3, s=smoothing)
    except Exception as e:
        print(f"Error al generar Spline: {e}")
        return pts

    # Remuestrear a la distancia exacta solicitada
    new_t = np.arange(0, total_length, separation_distance)
    
    # Asegurar que el punto final esté incluido
    if new_t[-1] != total_length:
        new_t = np.append(new_t, total_length)

    smooth_x = spl_x(new_t)
    smooth_y = spl_y(new_t)

    return np.vstack((smooth_x, smooth_y)).T

# ===================== 4. GUARDAR CSV =====================
def save_path_as_csv(path_points, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for p in path_points:
            writer.writerow([f"{p[0]:.4f}", f"{p[1]:.4f}"])
    print(f"--> Guardado: {filename} ({len(path_points)} puntos)")

# ===================== MAIN =====================
if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    HERE = Path(__file__).resolve().parent
    # Ajusta la ruta a tu mapa
    yaml_path = HERE.parent / "Mapas-F1Tenth" / "Oschersleben_map.yaml" 
    
    downsample_factor = 6
    x_start, y_start = -21.0, -4.0
    x_goal,  y_goal  = -19.0, -4.7

    # 1. Cargar Mapa (Obtenemos ruta imagen original para centrado)
    map_bin, rrt_resolution, rrt_origin, raw_img_path = load_map(yaml_path, downsample_factor)
    env = map_from_binary(map_bin)

    # Convertir start/goal a indices de grilla RRT
    start_node = world_to_map(x_start, y_start, rrt_resolution, rrt_origin)
    goal_node  = world_to_map(x_goal, y_goal, rrt_resolution, rrt_origin)

    print(f"Start: {start_node}, Goal: {goal_node}")
    
    # 2. Ejecutar RRT (Tu planificador original)
    print("Ejecutando RRT...")
    planner = SearchFactory()(
        "rrt",
        start=start_node,
        goal=goal_node,
        env=env,
        max_dist=4,       # Paso del RRT
        sample_num=100000 
    )
    planner.run()
    cost, path_indices, _ = planner.plan()

    if not path_indices:
        print("FALLO: RRT no encontró camino.")
    else:
        print(f"RRT Encontrado: {len(path_indices)} nodos.")
        
        # 3. Convertir ruta RRT a coordenadas de mundo (Raw Path)
        raw_world_path = []
        for xm, ym in path_indices:
            xw, yw = map_to_world(xm, ym, rrt_resolution, rrt_origin)
            raw_world_path.append([xw, yw])

        # Leer configuración original del YAML para el centrado preciso
        with open(yaml_path, 'r') as f:
            full_map_cfg = yaml.safe_load(f)
        
        # 4. CENTRADO (Distance Transform)
        # Esto empuja los puntos hacia el centro de la pista para evitar choques
        print("Aplicando centrado de trayectoria...")
        centered_path = get_centered_path(
            raw_world_path, 
            raw_img_path, 
            full_map_cfg['resolution'], 
            full_map_cfg['origin']
        )

        # 5. SUAVIZADO Y REMUESTREO (Cumpliendo la Tarea)
        # Tarea pide: Separación 1.0 metros
        print("Generando curva suavizada (1.0m)...")
        # smoothing=5.0 es un valor medio, auméntalo si la curva vibra mucho
        path_smooth_05 = apply_spline_and_resample(centered_path, separation_distance=1.0, smoothing=5.0)
        save_path_as_csv(path_smooth_05, "trayectoria_suavizada_1.0m.csv")

        # Tarea pide: Separación 1.0 metro
        print("Generando curva suavizada (1.0m)...")
        path_smooth_10 = apply_spline_and_resample(centered_path, separation_distance=1.0, smoothing=5.0)
        save_path_as_csv(path_smooth_10, "trayectoria_suavizada_1.0m.csv")

        # 6. VISUALIZACIÓN (Opcional, para verificar)
        img_viz = cv2.imread(raw_img_path)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
        h, w, _ = img_viz.shape
        res = full_map_cfg['resolution']
        org = full_map_cfg['origin']

        # Función auxiliar para plotear
        def to_pix(pts):
            pxs = []
            for p in pts:
                px = int((p[0] - org[0]) / res)
                py = int(h - (p[1] - org[1]) / res)
                pxs.append([px, py])
            return np.array(pxs)

        orig_px = to_pix(raw_world_path)
        cent_px = to_pix(centered_path)
        smooth_px = to_pix(path_smooth_05)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_viz)
        plt.plot(orig_px[:,0], orig_px[:,1], 'r.', label='RRT Crudo', alpha=0.4)
        plt.plot(cent_px[:,0], cent_px[:,1], 'bx', label='Centrado', alpha=0.5)
        plt.plot(smooth_px[:,0], smooth_px[:,1], 'g-', linewidth=2, label='Spline Final (1.0m)')
        plt.legend()
        plt.title("Generación de Curvas: RRT -> Centrado -> Spline")
        plt.show()
