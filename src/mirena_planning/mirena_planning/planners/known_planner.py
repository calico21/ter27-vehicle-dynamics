import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mirena_common.msg import EntityList, Entity, Gate, Track
from mirena_planning.planners.abstract_planner import BasePlanner, Plan
from geometry_msgs.msg import Vector3

class KnownPlanner(BasePlanner):
    def __init__(self, track_path=None):
        # Configuración inicial
        self.is_track_closed = False
        self.map_points = np.empty((0, 2))
        self.map_track = np.empty((0, 2))
        self.map_classes = np.empty(0)
        self.map_by_class = {}
        
        if track_path:
            self._load_json_map(track_path)
        
        self.map_tree = KDTree(self.map_points)

    def _load_json_map(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        
        # Carga eficiente de conos
        cones = data.get("cones", [])
        self.map_points = np.array([[c["x"], c["y"]] for c in cones], dtype=float)
        self.map_classes = np.array([c["type"] for c in cones])
        self.map_track = np.array([[c["x"], c["y"]] for c in data["path"]["points"]], dtype=float)
        
        # Metadata del path
        self.is_track_closed = data.get("path", {}).get("closed", False)
        
        # Agrupación por clase para RANSAC
        for cls in np.unique(self.map_classes):
            self.map_by_class[cls] = self.map_points[self.map_classes == cls]

    def plan(self, entity_list: EntityList) -> Plan | None:
        """Mantiene la interfaz original: Recibe EntityList y devuelve Track."""
        if not entity_list.entities:
            return None

        # Convertir mensaje a NumPy en un paso
        scan_pts = np.array([[e.position.x, e.position.y] for e in entity_list.entities])
        scan_cls = np.array([e.type for e in entity_list.entities])

        # Estimar transformación
        R, t, confidence = self.estimate_transform(scan_pts, scan_cls)

        if R is None or confidence < 0.9:
            return None

        # Transformar el mapa global al frame local del vehículo
        # Ecuación: Local = R.T @ (Global - t)
        transformed_track = (R.T @ (self.map_track - t).T).T
        ret_track =  self._build_track_msg(transformed_track)
        start_point_msg = Vector3()
        end_point_msg = Vector3()
        start_node = transformed_track[0]
        end_node = transformed_track[-1]
        start_point_msg.x, start_point_msg.y, start_point_msg.z = start_node[0], start_node[1], 0.0
        end_point_msg.x, end_point_msg.y, end_point_msg.z = end_node[0], end_node[1], 0.0
        return Plan(ret_track, start_point_msg, end_point_msg)

    def estimate_transform(self, scan_pts, scan_cls, threshold=0.9, inlier_dist=0.1, max_iters=400):
        if len(scan_pts) < 3:
            return None, None, 0.0

        best_score = 0.0
        best_R, best_t = None, None

        for _ in range(max_iters):
            # 1. Seleccionar tríptico aleatorio del scan
            idx = np.random.choice(len(scan_pts), 3, replace=False)
            triplet_scan = scan_pts[idx]
            triplet_cls = scan_cls[idx]

            # 2. Buscar correspondencias en el mapa por clase
            try:
                triplet_map = np.array([
                    self.map_by_class[c][np.random.choice(len(self.map_by_class[c]))] 
                    for c in triplet_cls
                ])
            except KeyError: continue # Salta si el scan ve un color que no está en el mapa

            # 3. Calcular transformación inicial (SVD)
            R, t = self.compute_rigid_transform(triplet_scan, triplet_map)
            
            # 4. Validar inliers (Geometría + Color)
            transformed = (R @ scan_pts.T).T + t
            dists, indices = self.map_tree.query(transformed)
            
            # Máscara de inliers: cercanía física Y mismo color
            inliers_mask = (dists < inlier_dist) & (self.map_classes[indices] == scan_cls)
            num_inliers = np.sum(inliers_mask)
            
            if num_inliers >= 3:
                # 5. Refinamiento opcional con todos los inliers encontrados
                R_ref, t_ref = self.compute_rigid_transform(scan_pts[inliers_mask], self.map_points[indices[inliers_mask]])
                
                score = num_inliers / len(scan_pts)
                if score > best_score:
                    best_score = score
                    best_R, best_t = R_ref, t_ref
                
                if best_score >= threshold:
                    break

        return best_R, best_t, best_score

    def compute_rigid_transform(self, A, B):
        """Implementación SVD estándar para alineación de puntos."""
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        
        H = (A - centroid_A).T @ (B - centroid_B)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Manejo de reflexión
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t

    def _build_track_msg(self, points) -> Track:
        """Helper para encapsular la creación del mensaje ROS."""
        track = Track()
        track.is_closed = self.is_track_closed
        for p in points:
            gate = Gate(x=p[0], y=p[1], psi=0.0)
            track.gates.append(gate)
        return track

    def debug_results(self, R, t, scan_pts, scan_cls):
        """Visualización de depuración (Matplotlib)."""
        if R is None: return
        
        transformed_scan = (R @ scan_pts.T).T + t
        plt.figure(figsize=(10, 7))
        
        # Dibujar mapa global
        plt.scatter(self.map_points[:, 0], self.map_points[:, 1], c='gray', alpha=0.3, label='Global Map')
        
        # Dibujar scan transformado por colores
        color_map = {'cone_blue': 'blue', 'cone_yellow': 'yellow', 'cone_orange': 'orange', 'cone_big_orange': 'red'}
        for cls in np.unique(scan_cls):
            mask = scan_cls == cls
            plt.scatter(transformed_scan[mask, 0], transformed_scan[mask, 1], 
                        c=color_map.get(cls, 'black'), marker='x', label=f'Scan {cls}')

        plt.title(f"RANSAC Alignment Results")
        plt.legend()
        plt.axis('equal')
        plt.show()