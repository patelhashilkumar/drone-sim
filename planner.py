"""
=============================================================================
  MIGRATION NOTE — RRT* replaces A* (March 2026)
=============================================================================
  Previous:  26-directional grid-based A* search in 3D (integer coordinates).
  Current :  RRT* (Rapidly-exploring Random Tree Star) in continuous 3D space.

  WHY?
  • A* is optimal on discrete grids but locked to integer steps. RRT* explores
    continuous space natively, producing smoother pre-spline waypoints.
  • RRT* supports an energy-aware cost function (distance + altitude gain +
    wind drag) which A* cannot express without expensive grid inflation.
  • RRT*'s asymptotic optimality guarantee means it converges toward the true
    shortest energy-optimal path as iterations increase.

  WHAT STAYED THE SAME?
  • Constructor signature:  HybridPlanner(environment, drone_id)
  • Public API:            planner.plan(num_points) → (path, metrics)
  • B-Spline smoothing step (scipy splprep/splev) is untouched.
  • Telemetry output format: {distance, energy, fitness, collisions}
  • Full compatibility with server.py WebSocket stream and index.html frontend.
=============================================================================
"""

import numpy as np
import math
from scipy.interpolate import splprep, splev


# ═══════════════════════════════════════════════════════════════════════════════
#  ENERGY COST FUNCTION  (configurable α, β, γ weights)
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyCost:
    """
    Computes the traversal cost between two 3D points using a weighted
    combination of distance, altitude gain, and simulated wind drag.

    cost = α × euclidean_distance
         + β × max(0, Δz)           # penalise climbing, not descending
         + γ × horizontal_distance  # proxy for wind drag in open air

    Attributes:
        alpha (float): Weight for straight-line distance.
        beta  (float): Weight for altitude gain (climbing penalty).
        gamma (float): Weight for wind drag (horizontal exposure).
    """

    def __init__(self, alpha=1.0, beta=0.8, gamma=0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, p1, p2):
        """Return the energy cost of moving from p1 to p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        euclidean   = math.sqrt(dx*dx + dy*dy + dz*dz)
        alt_gain    = max(0.0, dz)                      # only penalise climbing
        horiz_dist  = math.sqrt(dx*dx + dy*dy)           # wind drag proxy

        return (self.alpha * euclidean
              + self.beta  * alt_gain
              + self.gamma * horiz_dist)


# ═══════════════════════════════════════════════════════════════════════════════
#  RRT* NODE
# ═══════════════════════════════════════════════════════════════════════════════

class RRTNode:
    """Single node in the RRT* tree."""
    __slots__ = ('pos', 'parent', 'cost', 'children')

    def __init__(self, pos, parent=None, cost=0.0):
        self.pos = np.array(pos, dtype=np.float64)   # (x, y, z)
        self.parent = parent                          # RRTNode | None
        self.cost = cost                              # cumulative cost from start
        self.children = []


# ═══════════════════════════════════════════════════════════════════════════════
#  HYBRID PLANNER  —  RRT* global search  +  B-Spline local smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class HybridPlanner:
    """
    RRT* + B-Spline Path Planner.

    Args:
        environment: Reference to the Environment instance (grid, drones, etc.).
        drone_id:    Which drone entry to plan for.
    """

    def __init__(self, environment, drone_id):
        self.env = environment
        self.drone_id = drone_id

        self.start = tuple(self.env.drones[drone_id]['start'])
        self.goal  = tuple(self.env.drones[drone_id]['goal'])

        # RRT* hyper-parameters
        self.max_iter    = 2000    # maximum random samples
        self.step_size   = 1.5     # steer distance per extension (metres)
        self.search_rad  = 3.0     # near-neighbour rewire radius
        self.goal_thresh = 2.0     # how close to goal counts as "reached"

        # 3D bounds (continuous)
        self.x_range = (0, self.env.width)
        self.y_range = (0, self.env.height)
        self.z_range = (1, self.env.max_altitude - 1)   # stay above ground, below ceiling

        # Energy cost function  (tunable)
        self.cost_fn = EnergyCost(alpha=1.0, beta=0.8, gamma=0.3)

        # Legacy energy model for final telemetry (kept for UI compatibility)
        self.base_energy_rate      = 0.1
        self.turning_energy_factor = 0.3
        self.hover_energy_rate     = 0.05

    # ── RRT* core methods ─────────────────────────────────────────────────

    def _sample(self):
        """
        sample() — draw a random point in the 3D configuration space.
        With 10 % probability, return the goal directly (goal bias)
        to accelerate convergence.
        """
        if np.random.random() < 0.10:
            return np.array(self.goal, dtype=np.float64)

        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        z = np.random.uniform(*self.z_range)
        return np.array([x, y, z])

    def _nearest(self, tree, point):
        """
        nearest() — find the node in *tree* closest to *point*
        using vectorized Euclidean distance.
        """
        pos_array = np.array([n.pos for n in tree])
        dists = np.linalg.norm(pos_array - point, axis=1)
        best_idx = np.argmin(dists)
        return tree[best_idx]

    def _steer(self, from_pos, to_pos):
        """
        steer() — move from *from_pos* toward *to_pos* by at most
        self.step_size.  Returns the new position as an ndarray.
        """
        diff = to_pos - from_pos
        dist = np.linalg.norm(diff)
        if dist <= self.step_size:
            return to_pos.copy()
        return from_pos + (diff / dist) * self.step_size

    def _collision_free(self, p1, p2, steps=10):
        """
        Check the straight segment p1→p2 for collisions by sampling
        *steps* equally-spaced points along it.
        """
        for t in np.linspace(0, 1, steps):
            pt = p1 + (p2 - p1) * t
            ix, iy, iz = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
            if self.env.is_collision(ix, iy, iz):
                return False
        return True

    def _near(self, tree, point):
        """
        near() — return all nodes within self.search_rad of *point*
        using vectorized Euclidean distance.
        """
        pos_array = np.array([n.pos for n in tree])
        dists = np.linalg.norm(pos_array - point, axis=1)
        indices = np.where(dists <= self.search_rad)[0]
        return [tree[i] for i in indices]

    def _rewire(self, tree, new_node, neighbours):
        """
        rewire() — for every neighbour, check whether routing
        *through* new_node is cheaper than the neighbour's current
        parent.  If so, re-parent.

        This is the key step that makes RRT* asymptotically optimal
        (plain RRT does not do this).
        """
        for nb in neighbours:
            if nb is new_node.parent:
                continue
            potential_cost = new_node.cost + self.cost_fn(new_node.pos, nb.pos)
            if potential_cost < nb.cost and self._collision_free(new_node.pos, nb.pos):
                # Re-parent
                if nb.parent:
                    nb.parent.children.remove(nb)
                nb.parent = new_node
                nb.cost = potential_cost
                new_node.children.append(nb)

    def _extract_path(self, goal_node):
        """Walk backwards from goal_node to root and return waypoints."""
        path = []
        node = goal_node
        while node is not None:
            path.append(tuple(node.pos))
            node = node.parent
        return path[::-1]

    # ── main search routine ───────────────────────────────────────────────

    def find_rrtstar_path(self):
        """
        Runs the RRT* algorithm in continuous 3D space.

        Returns:
            list[(x, y, z)]  —  raw waypoints from start to goal, or None.
        """
        root = RRTNode(self.start, parent=None, cost=0.0)
        tree = [root]

        best_goal_node = None
        best_goal_cost = float('inf')

        for _ in range(self.max_iter):
            # 1. Sample a random configuration
            q_rand = self._sample()

            # 2. Find nearest existing node
            q_nearest = self._nearest(tree, q_rand)

            # 3. Steer towards the sample
            q_new_pos = self._steer(q_nearest.pos, q_rand)

            # 4. Collision check on the segment
            if not self._collision_free(q_nearest.pos, q_new_pos):
                continue

            # 5. Choose the best parent from neighbours (RRT* improvement)
            neighbours = self._near(tree, q_new_pos)
            best_parent = q_nearest
            best_cost   = q_nearest.cost + self.cost_fn(q_nearest.pos, q_new_pos)

            for nb in neighbours:
                c = nb.cost + self.cost_fn(nb.pos, q_new_pos)
                if c < best_cost and self._collision_free(nb.pos, q_new_pos):
                    best_parent = nb
                    best_cost   = c

            # 6. Insert the new node
            new_node = RRTNode(q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(new_node)
            tree.append(new_node)

            # 7. Rewire nearby nodes through the new node
            self._rewire(tree, new_node, neighbours)

            # 8. Check if we reached the goal
            dist_to_goal = np.linalg.norm(q_new_pos - np.array(self.goal))
            if dist_to_goal <= self.goal_thresh:
                total = new_node.cost + self.cost_fn(new_node.pos, np.array(self.goal))
                if total < best_goal_cost:
                    # Connect to exact goal
                    if self._collision_free(new_node.pos, np.array(self.goal)):
                        goal_node = RRTNode(self.goal, parent=new_node,
                                            cost=total)
                        new_node.children.append(goal_node)
                        tree.append(goal_node)
                        best_goal_node = goal_node
                        best_goal_cost = total

        if best_goal_node is None:
            return None

        return self._extract_path(best_goal_node)

    # ── B-Spline smoothing (unchanged from A* era) ───────────────────────

    def smooth_trajectory(self, raw_path, num_points=120, spline_degree=3, spline_smoothness=3.0):
        """
        Applies mathematical B-Spline interpolation to the raw RRT* waypoints.

        Args:
            raw_path:           List of (x,y,z) waypoints from RRT*.
            num_points:         Number of interpolated output points.
            spline_degree:      B-Spline degree k (1=linear, 3=cubic, 5=quintic).
            spline_smoothness:  Smoothing factor s for splprep.
        """
        if not raw_path or len(raw_path) < 2:
            return raw_path

        x = [p[0] for p in raw_path]
        y = [p[1] for p in raw_path]
        z = [p[2] for p in raw_path]

        # Very short paths → simple linear interpolation
        if len(raw_path) <= 3:
            pts = np.column_stack([
                np.linspace(x[0], x[-1], num_points),
                np.linspace(y[0], y[-1], num_points),
                np.linspace(z[0], z[-1], num_points)
            ])
            return pts.tolist()

        # Fit B-Spline in 3D with configurable degree and smoothness
        k = min(spline_degree, len(x) - 1)
        try:
            tck, _u = splprep([x, y, z], s=spline_smoothness, k=k)
            u_new = np.linspace(0, 1.0, num_points)
            x_new, y_new, z_new = splev(u_new, tck)
            return list(zip(x_new, y_new, z_new))
        except ValueError:
            return raw_path

    # ── Telemetry  (unchanged — keeps UI parity) ────────────────────────

    def calculate_metrics(self, smoothed_path):
        """Calculates distance, energy consumption, and collision counts."""
        if not smoothed_path or len(smoothed_path) < 2:
            return 0.0, 0.0, 999

        path = np.array(smoothed_path)

        # Distance
        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        distance = float(np.sum(segment_lengths))

        # Turn-based energy (angles between consecutive segments)
        turn_energy = 0.0
        if len(diffs) >= 2:
            v1 = diffs[:-1]
            v2 = diffs[1:]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            valid = (n1 > 1e-10) & (n2 > 1e-10)
            if np.any(valid):
                dots = np.sum(v1[valid] * v2[valid], axis=1)
                cos_a = np.clip(dots / (n1[valid] * n2[valid]), -1, 1)
                turn_energy = float(np.sum(np.arccos(cos_a)) * self.turning_energy_factor)

        distance_energy = distance * self.base_energy_rate
        hover_energy    = (len(path) - 1) * self.hover_energy_rate
        total_energy    = distance_energy + turn_energy + hover_energy

        # Collision audit along the smoothed spline
        collisions = 0
        for i in range(len(path) - 1):
            for t in np.linspace(0, 1, 5):
                px = path[i][0] + (path[i+1][0] - path[i][0]) * t
                py = path[i][1] + (path[i+1][1] - path[i][1]) * t
                pz = path[i][2] + (path[i+1][2] - path[i][2]) * t
                if self.env.is_collision(int(round(px)), int(round(py)), int(round(pz))):
                    collisions += 1

        return distance, total_energy, collisions

    # ── Public API  (signature identical to the old A* version) ──────────

    def plan(self, num_points=120, spline_degree=3, spline_smoothness=3.0):
        """
        Executes:  RRT* global search  →  B-Spline local smoothing  →  Telemetry.

        Returns:
            best_path (list[tuple]): Smoothed high-resolution [x, y, z] waypoints.
            metrics   (dict):        {distance, energy, fitness, collisions}
        """
        # 1. RRT* global search
        raw_path = self.find_rrtstar_path()

        # Fallback if no solution found
        if not raw_path:
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(self.start, self.goal)))
            return [self.start, self.goal], {
                'fitness': 999999, 'energy': 999999,
                'distance': dist,  'collisions': 999
            }

        # 2. B-Spline trajectory smoothing
        aerodynamic_path = self.smooth_trajectory(
            raw_path, num_points,
            spline_degree=spline_degree,
            spline_smoothness=spline_smoothness
        )

        # 3. Telemetry
        distance, energy, collisions = self.calculate_metrics(aerodynamic_path)
        metrics = {
            'distance':   float(distance),
            'energy':     float(energy),
            'fitness':    float(distance + energy * 2 + collisions * 10000),
            'collisions': int(collisions)
        }
        return aerodynamic_path, metrics
