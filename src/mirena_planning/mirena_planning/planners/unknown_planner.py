import numpy as np
import math
from scipy.spatial import Delaunay, KDTree
from mirena_planning.planners.abstract_planner import BasePlanner, Plan
from mirena_common.msg import EntityList, Gate, Track
from geometry_msgs.msg import Vector3

class UnknownPlanner(BasePlanner):
    # Constants for easier tuning
    CLOSURE_THRESHOLD = 8.0
    MIN_PATH_POINTS = 15
    MAX_GATE_GAP = 8.0        # max spacing between consecutive centerline midpoints
    MAX_GATE_WIDTH = 4.5      # reject blue<->yellow Delaunay edges wider than a plausible gate
    TURN_PENALTY = 1.5        # how strongly the trace prefers heading continuity over proximity
    MIN_PROGRESS_COS = 0.1    # require real forward progress (reject turns sharper than ~84deg)
    
    def __init__(self):
        super().__init__()
        self.blue_cones = np.empty((0, 2))
        self.yellow_cones = np.empty((0, 2))
        self.orange_cones = np.empty((0, 2))
        self.is_track_closed = False
        self.start_point = Vector3()
        

    def plan(self, entity_list: EntityList) -> Plan | None:
        """Main entry point: takes EntityList, returns Track."""
        # 1. Parse Entities
        self._extract_cones(entity_list)

        return self._compute_path_logic()


    def _extract_cones(self, entity_list: EntityList):
        """Filters the EntityList into color-coded numpy arrays."""
        blue, yellow, orange = [], [], []
        for e in entity_list.entities:
            pos = [e.position.x, e.position.y]
            color = e.type.lower()
            if 'blue' in color: blue.append(pos)
            elif 'yellow' in color: yellow.append(pos)
            elif 'orange' in color: orange.append(pos)
        
        self.blue_cones = np.array(blue) if blue else np.empty((0, 2))
        self.yellow_cones = np.array(yellow) if yellow else np.empty((0, 2))
        self.orange_cones = np.array(orange) if orange else np.empty((0, 2))

    def _compute_path_logic(self) -> Plan | None:
        """Geometry logic with safety guards to prevent SIGABRT."""
        # Safety Guard: Need 2 orange (start/finish) and some track boundaries
        if len(self.orange_cones) < 2 or (len(self.blue_cones) == 0 and len(self.yellow_cones) == 0):
            return None

        # 1. Start Node
        start_node = np.mean(self.orange_cones, axis=0)

        # 2. Triangulate Midpoints
        all_cones = np.vstack((self.blue_cones, self.yellow_cones))
        num_blue = len(self.blue_cones)
        
        if len(all_cones) < 3:
            return None

        try:
            tri = Delaunay(all_cones)
        except Exception: # Handle collinear points or geometric errors
            return None

        midpoints = []
        for simplex in tri.simplices:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                a, b = simplex[i], simplex[j]
                # Boundary connection: one point is blue, other is yellow
                if (a < num_blue) != (b < num_blue):
                    # Only keep edges narrow enough to be a real track gate. This drops
                    # spurious long diagonals (e.g. across a hairpin) that derail the trace.
                    if np.linalg.norm(all_cones[a] - all_cones[b]) <= self.MAX_GATE_WIDTH:
                        midpoints.append((all_cones[a] + all_cones[b]) / 2)

        if not midpoints:
            return None

        # Round before unique to merge near-coincident midpoints from adjacent triangles.
        midpoints = np.unique(np.round(np.array(midpoints), 2), axis=0)
        mid_tree = KDTree(midpoints)

        # 3. Initial heading from cone colours (FS: blue on the left, yellow on the right)
        forward = self._estimate_forward(start_node)

        # 4. Trace the centerline with a direction-aware walk along the midpoints
        path = self._trace_centerline(midpoints, mid_tree, start_node, forward)
        if len(path) < 2:
            return None

        # 6. Loop Closure (only if the path travelled away from the start and returned)
        self.is_track_closed = self._detect_closure(path, start_node)

        # Drop a trailing point that coincides with the start so the downstream periodic
        # spline doesn't get a duplicate (zero-length) seam segment.
        if self.is_track_closed and len(path) > 2 and np.linalg.norm(path[-1] - start_node) < 1.0:
            path = path[:-1]


        # Construct the plan and return it
        ret_track =  self._build_track_msg(path)
        start_point_msg = Vector3()
        start_point_msg.x, start_point_msg.y, start_point_msg.z = start_node[0], start_node[1], 0.0
        self.start_point = start_point_msg
        ret_start_point = start_point_msg
        ret_end_point = start_point_msg # If closed we assume its the same one lmao
        return Plan(ret_track, ret_start_point, ret_end_point)
    
    def _estimate_forward(self, start_node):
        """Unit vector of the correct driving direction inferred from cone colours.

        FS convention: blue cones are on the driver's left, yellow on the right, so the
        vector blue->yellow points to the right of travel. Rotating it +90deg (CCW) yields
        'forward'. Returns None if both colours aren't available to disambiguate.
        """
        if len(self.blue_cones) == 0 or len(self.yellow_cones) == 0:
            return None

        _, b_idx = KDTree(self.blue_cones).query(start_node)
        _, y_idx = KDTree(self.yellow_cones).query(start_node)

        right = self.yellow_cones[y_idx] - self.blue_cones[b_idx]
        forward = np.array([-right[1], right[0]])  # rotate +90deg
        norm = np.linalg.norm(forward)
        return forward / norm if norm > 1e-6 else None

    def _select_seed(self, midpoints, mid_tree, start_node, forward):
        """First midpoint to follow: the nearby midpoint best aligned with 'forward'.

        Considers the closest handful of midpoints (not just two) and picks the one whose
        bearing from the start gate best matches the desired heading. Falls back to the
        nearest midpoint when direction can't be inferred.
        """
        k = min(len(midpoints), 5)
        dists, idxs = mid_tree.query(start_node, k=k)
        idxs = np.atleast_1d(idxs)

        if forward is None:
            return int(idxs[0])

        best_idx, best_score = int(idxs[0]), -np.inf
        for idx in idxs:
            v = midpoints[idx] - start_node
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                continue
            proj = float(np.dot(v / norm, forward))  # cosine of bearing vs desired heading
            if proj > best_score:
                best_score, best_idx = proj, int(idx)
        return best_idx

    def _detect_closure(self, path, start_node):
        """True only if the path left the start region and came back.

        Guards against declaring a closed lap on a short stub that never leaves the start
        line, which the previous final-point-only check could do.
        """
        if len(path) <= self.MIN_PATH_POINTS:
            return False

        dist_from_start = np.linalg.norm(path - start_node, axis=1)
        returned = dist_from_start[-1] < self.CLOSURE_THRESHOLD
        went_away = float(np.max(dist_from_start)) > 2.0 * self.CLOSURE_THRESHOLD
        return bool(returned and went_away)

    def _trace_centerline(self, midpoints, mid_tree, start_node, forward):
        """Order the midpoints into a centerline using proximity + heading continuity.

        Unlike a plain nearest-neighbour walk, each step picks the unvisited midpoint that
        is both close and roughly straight ahead of the current heading, and it never steps
        backwards. This avoids cross-track jumps, hairpin shortcuts, and the early
        termination that bites greedy NN when the few nearest points are all visited.
        """
        n = len(midpoints)
        if n == 0:
            return np.empty((0, 2))

        seed = self._select_seed(midpoints, mid_tree, start_node, forward)
        order = [seed]
        remaining = set(range(n))
        remaining.discard(seed)

        # Initial heading: from the start gate toward the seed (fall back to colour forward).
        heading = midpoints[seed] - start_node
        if np.linalg.norm(heading) < 1e-6:
            heading = forward if forward is not None else np.array([1.0, 0.0])
        heading = heading / (np.linalg.norm(heading) + 1e-9)

        while remaining:
            curr = midpoints[order[-1]]
            best_idx, best_cost, best_dir = -1, np.inf, None
            for idx in remaining:
                vec = midpoints[idx] - curr
                dist = float(np.linalg.norm(vec))
                if dist > self.MAX_GATE_GAP or dist < 1e-6:
                    print(dist)
                    continue
                direction = vec / dist
                cos_turn = float(np.dot(direction, heading))
                if cos_turn < self.MIN_PROGRESS_COS:   # too sharp / backwards
                    continue
                cost = dist * (1.0 + self.TURN_PENALTY * (1.0 - cos_turn))
                if cost < best_cost:
                    best_cost, best_idx, best_dir = cost, idx, direction
            if best_idx < 0:
                break
            order.append(best_idx)
            remaining.discard(best_idx)
            heading = best_dir

        return np.array([start_node] + [midpoints[i] for i in order])

    def _build_track_msg(self, points: np.ndarray) -> Track:
        """Converts numpy points to Track message with proper Gate headings."""
        track = Track()
        track.is_closed = self.is_track_closed
        
        n = len(points)
        for i in range(n):
            # Calculate psi (heading) toward next point
            if i < n - 1:
                dx = points[i+1][0] - points[i][0]
                dy = points[i+1][1] - points[i][1]
                psi = math.atan2(dy, dx)
            elif self.is_track_closed:
                dx = points[0][0] - points[i][0]
                dy = points[0][1] - points[i][1]
                psi = math.atan2(dy, dx)
            else:
                # Last point keeps previous orientation
                psi = track.gates[-1].psi if track.gates else 0.0

            gate = Gate()
            gate.x, gate.y, gate.psi = float(points[i][0]), float(points[i][1]), float(psi)
            track.gates.append(gate)

        return track