import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from fastapi.staticfiles import StaticFiles

from environment import Environment
from planner import HybridPlanner

logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(title="Drone RRT* Path Planner API")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 50, 50
DRONES_CONFIG = [
    {'id': 0, 'name': 'ALPHA',   'start': (5, 5, 1),   'goal': (45, 45, 1)},
    {'id': 1, 'name': 'BRAVO',   'start': (5, 45, 1),  'goal': (45, 5, 1)},
    {'id': 2, 'name': 'CHARLIE', 'start': (25, 5, 1),  'goal': (25, 45, 1)},
]

SIMULATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulations")

# Per-connection session state
sessions = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION STORE  —  JSON file-based persistence
# ═══════════════════════════════════════════════════════════════════════════════
class SimulationStore:
    """Persists simulation results as individual JSON files in simulations/."""

    def __init__(self, directory: str = SIMULATIONS_DIR):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _filepath(self, sim_id: str) -> str:
        return os.path.join(self.directory, f"{sim_id}.json")

    def save(self, env, params, results, algorithm, planning_time):
        """Save a completed simulation to disk. Returns the record dict."""
        sim_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc)

        # Extract obstacles snapshot
        obs_y, obs_x = np.where(env.grid > 0)
        obstacles = [
            {'x': int(x), 'y': int(y), 'h': int(env.grid[y, x])}
            for x, y in zip(obs_x.tolist(), obs_y.tolist())
        ]

        # Build per-drone data
        drones_data = []
        total_distance = 0.0
        total_energy = 0.0
        total_collisions = 0
        for drone_id, data in results.items():
            cfg = next(d for d in DRONES_CONFIG if d['id'] == drone_id)
            path_list = np.array(data['path']).tolist() if isinstance(data['path'], (list, np.ndarray)) else data['path']
            drones_data.append({
                'id': drone_id,
                'name': cfg['name'],
                'start': list(cfg['start']),
                'goal': list(cfg['goal']),
                'path': path_list,
                'metrics': data['metrics']
            })
            total_distance += data['metrics'].get('distance', 0)
            total_energy += data['metrics'].get('energy', 0)
            total_collisions += data['metrics'].get('collisions', 0)

        record = {
            'id': sim_id,
            'timestamp': now.isoformat(),
            'algorithm': algorithm,
            'params': params.copy(),
            'planning_time': round(planning_time, 3),
            'grid': {'width': WIDTH, 'height': HEIGHT},
            'obstacles': obstacles,
            'drones': drones_data,
            'summary': {
                'total_distance': round(total_distance, 2),
                'total_energy': round(total_energy, 2),
                'total_collisions': total_collisions,
                'avg_fitness': round(
                    sum(d['metrics'].get('fitness', 0) for d in drones_data) / max(len(drones_data), 1), 2
                ),
                'drone_count': len(drones_data)
            }
        }

        with open(self._filepath(sim_id), 'w') as f:
            json.dump(record, f, indent=2)

        logger.info(f"Simulation {sim_id} saved ({len(drones_data)} drones, {planning_time:.2f}s)")
        return record

    def list_all(self):
        """Return a list of simulation summaries (without full path data)."""
        entries = []
        if not os.path.exists(self.directory):
            return entries
        for fname in sorted(os.listdir(self.directory), reverse=True):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.directory, fname), 'r') as f:
                    data = json.load(f)
                entries.append({
                    'id': data['id'],
                    'timestamp': data['timestamp'],
                    'algorithm': data['algorithm'],
                    'planning_time': data.get('planning_time', 0),
                    'summary': data.get('summary', {}),
                    'params': data.get('params', {})
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return entries

    def get(self, sim_id: str):
        """Return full simulation data for a given ID, or None."""
        fp = self._filepath(sim_id)
        if not os.path.exists(fp):
            return None
        with open(fp, 'r') as f:
            return json.load(f)

    def delete(self, sim_id: str) -> bool:
        """Delete a saved simulation. Returns True if deleted."""
        fp = self._filepath(sim_id)
        if os.path.exists(fp):
            os.remove(fp)
            logger.info(f"Simulation {sim_id} deleted")
            return True
        return False


store = SimulationStore()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_environment():
    """Create a fresh Environment with city blocks."""
    env = Environment(WIDTH, HEIGHT)
    for d in DRONES_CONFIG:
        env.add_drone(d['id'], d['start'], d['goal'])
    env.generate_city_blocks()
    return env


def extract_obstacles(env):
    """Pull obstacle list from environment grid."""
    obs_y, obs_x = np.where(env.grid > 0)
    return [
        {'x': int(x), 'y': int(y), 'h': int(env.grid[y, x])}
        for x, y in zip(obs_x.tolist(), obs_y.tolist())
    ]


def default_params():
    return {
        'spline_degree': 3,
        'spline_smoothness': 3.0,
        'output_resolution': 120,
        'heuristic_weight': 1,
        'algorithm': 'rrt_star'
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════════════════════
#  REST API  —  Simulation History
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/simulations")
async def list_simulations():
    """Return all saved simulations (summaries only)."""
    return JSONResponse(store.list_all())


@app.get("/api/simulations/{sim_id}")
async def get_simulation(sim_id: str):
    """Return full simulation data for the given ID."""
    data = store.get(sim_id)
    if data is None:
        return JSONResponse({"error": "Simulation not found"}, status_code=404)
    return JSONResponse(data)


@app.delete("/api/simulations/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a saved simulation."""
    if store.delete(sim_id):
        return JSONResponse({"status": "deleted", "id": sim_id})
    return JSONResponse({"error": "Simulation not found"}, status_code=404)


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    ws_id = id(websocket)

    # Build first map on connect
    env = build_environment()
    sessions[ws_id] = {'env': env, 'params': default_params()}
    await send_map(websocket, env)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command")
            except json.JSONDecodeError:
                cmd = data

            session = sessions.get(ws_id)
            if not session:
                continue

            # ── Start Simulation ──────────────────────────────────────
            if cmd == "start_simulation":
                active = msg.get("active_drones", None)
                await run_planning(websocket, session, active)

            # ── Randomize Map ─────────────────────────────────────────
            elif cmd == "randomize_map":
                env = build_environment()
                session['env'] = env
                await send_map(websocket, env)
                await ws_log(websocket, "New city generated. Click Start to optimize.")

            # ── Update Planner Parameters ─────────────────────────────
            elif cmd == "update_params":
                p = session['params']
                if 'heuristic_weight'  in msg: p['heuristic_weight']  = int(msg['heuristic_weight'])
                if 'spline_degree'     in msg: p['spline_degree']     = int(msg['spline_degree'])
                if 'spline_smoothness' in msg: p['spline_smoothness'] = float(msg['spline_smoothness'])
                if 'output_resolution' in msg: p['output_resolution'] = int(msg['output_resolution'])
                await ws_log(websocket, f"Params updated: {p}")

            # ── Switch Algorithm ──────────────────────────────────────
            elif cmd == "switch_algorithm":
                algo = msg.get("algorithm", "rrt_star")
                session['params']['algorithm'] = algo
                names = {'rrt_star': 'RRT*', 'astar': 'A*', 'dijkstra': 'Dijkstra'}
                await websocket.send_text(json.dumps({
                    "type": "algorithm_switched",
                    "algorithm": algo,
                    "label": names.get(algo, algo) + " + B-Spline Smoothing"
                }))
                await ws_log(websocket, f"Algorithm switched to {names.get(algo, algo)}")

            # ── Edit Building Height ──────────────────────────────────
            elif cmd == "edit_building":
                x, y, h = int(msg['x']), int(msg['y']), int(msg['h'])
                env = session['env']
                if 0 <= x < env.width and 0 <= y < env.height:
                    env.grid[y, x] = h
                    obstacles = extract_obstacles(env)
                    await websocket.send_text(json.dumps({
                        "type": "map_updated", "obstacles": obstacles
                    }))

            # ── Add Obstacle ──────────────────────────────────────────
            elif cmd == "add_obstacle":
                x, y, h = int(msg['x']), int(msg['y']), int(msg.get('h', 3))
                env = session['env']
                if 0 <= x < env.width and 0 <= y < env.height:
                    env.grid[y, x] = h

            # ── Remove Obstacle ───────────────────────────────────────
            elif cmd == "remove_obstacle":
                x, y = int(msg['x']), int(msg['y'])
                env = session['env']
                if 0 <= x < env.width and 0 <= y < env.height:
                    env.grid[y, x] = 0

            # ── Load Saved Simulation ─────────────────────────────────
            elif cmd == "load_simulation":
                sim_id = msg.get("simulation_id")
                if sim_id:
                    await load_saved_simulation(websocket, session, sim_id)

            # ── Compare Two Simulations ───────────────────────────────
            elif cmd == "compare_simulations":
                id_a = msg.get("id_a")
                id_b = msg.get("id_b")
                if id_a and id_b:
                    await compare_simulations(websocket, id_a, id_b)

            elif cmd == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        sessions.pop(ws_id, None)
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        sessions.pop(ws_id, None)
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
async def ws_log(ws, msg, source="SYSTEM"):
    await ws.send_text(json.dumps({"type": "log", "source": source, "msg": msg}))


async def send_map(websocket, env):
    obstacles = extract_obstacles(env)
    await websocket.send_text(json.dumps({
        "type": "init",
        "grid": {"width": WIDTH, "height": HEIGHT},
        "obstacles": obstacles,
        "drones": DRONES_CONFIG
    }))


async def run_planning(websocket, session, active_drones=None):
    env = session['env']
    p = session['params']

    # Tell frontend planning has begun
    await websocket.send_text(json.dumps({"type": "planning_start"}))
    await ws_log(websocket, "Starting RRT* + B-Spline Path Search...")

    # Determine which drones to plan for
    active_ids = active_drones if active_drones is not None else [d['id'] for d in DRONES_CONFIG]

    # Plan for each active drone (timed)
    t_start = time.time()
    results = {}
    for d in DRONES_CONFIG:
        if d['id'] not in active_ids:
            continue
        planner = HybridPlanner(environment=env, drone_id=d['id'])
        path, metrics = planner.plan(
            num_points=p['output_resolution'],
            spline_degree=p['spline_degree'],
            spline_smoothness=p['spline_smoothness']
        )
        results[d['id']] = {"path": path, "metrics": metrics}
    planning_time = time.time() - t_start

    # ── Auto-save to simulation store ─────────────────────────────
    saved_record = store.save(env, p, results, p.get('algorithm', 'rrt_star'), planning_time)
    sim_id = saved_record['id']

    # Tell frontend planning is done, animation begins
    await websocket.send_text(json.dumps({
        "type": "planning_done",
        "planning_time": round(planning_time, 2),
        "simulation_id": sim_id
    }))
    await ws_log(websocket, f"Optimal flight trajectories generated in {planning_time:.2f}s. Streaming to client..")

    # Stream animation frames
    num_points = p['output_resolution']
    max_frames = num_points
    for frame in range(max_frames):
        update = {"type": "update", "iteration": frame, "max_frames": max_frames, "drones": []}
        for drone_id, data in results.items():
            arr = np.array(data["path"])
            end = max(2, int((frame / max_frames) * len(arr)))
            update["drones"].append({
                "id": drone_id,
                "best_path": arr[:end].tolist(),
                "fitness": data["metrics"]["fitness"],
                "energy": data["metrics"]["energy"],
                "distance": data["metrics"]["distance"],
                "collisions": data["metrics"]["collisions"]
            })
        await websocket.send_text(json.dumps(update))
        await asyncio.sleep(0.04)

    # Send completion with summary
    await ws_log(websocket, "RRT* routing complete!")
    await websocket.send_text(json.dumps({
        "type": "complete",
        "simulation_id": sim_id,
        "planning_time": round(planning_time, 2),
        "summary": saved_record['summary']
    }))


async def load_saved_simulation(websocket, session, sim_id):
    """Load a past simulation and replay it on the frontend."""
    record = store.get(sim_id)
    if not record:
        await ws_log(websocket, f"Simulation {sim_id} not found.", "WARNING")
        return

    # Rebuild environment from saved obstacles
    env = Environment(WIDTH, HEIGHT)
    for d in DRONES_CONFIG:
        env.add_drone(d['id'], d['start'], d['goal'])
    for obs in record.get('obstacles', []):
        env.grid[obs['y'], obs['x']] = obs['h']
    session['env'] = env

    # Send the map
    await send_map(websocket, env)
    await ws_log(websocket, f"Loaded simulation {sim_id} from {record['timestamp'][:10]}")

    # Tell frontend planning is done (skip planning phase)
    await websocket.send_text(json.dumps({
        "type": "planning_done",
        "planning_time": record.get('planning_time', 0),
        "simulation_id": sim_id,
        "is_replay": True
    }))

    # Build results from saved data
    results = {}
    for drone_data in record.get('drones', []):
        results[drone_data['id']] = {
            "path": drone_data['path'],
            "metrics": drone_data['metrics']
        }

    # Stream animation frames (replay)
    num_points = record.get('params', {}).get('output_resolution', 120)
    max_frames = num_points
    for frame in range(max_frames):
        update = {"type": "update", "iteration": frame, "max_frames": max_frames, "drones": []}
        for drone_id, data in results.items():
            arr = np.array(data["path"])
            end = max(2, int((frame / max_frames) * len(arr)))
            update["drones"].append({
                "id": drone_id,
                "best_path": arr[:end].tolist(),
                "fitness": data["metrics"]["fitness"],
                "energy": data["metrics"]["energy"],
                "distance": data["metrics"]["distance"],
                "collisions": data["metrics"]["collisions"]
            })
        await websocket.send_text(json.dumps(update))
        await asyncio.sleep(0.04)

    await ws_log(websocket, f"Replay of simulation {sim_id} complete!")
    await websocket.send_text(json.dumps({
        "type": "complete",
        "simulation_id": sim_id,
        "is_replay": True,
        "summary": record.get('summary', {})
    }))


async def compare_simulations(websocket, id_a, id_b):
    """Send two simulation results to the frontend for side-by-side comparison."""
    rec_a = store.get(id_a)
    rec_b = store.get(id_b)
    if not rec_a or not rec_b:
        await ws_log(websocket, "One or both simulations not found.", "WARNING")
        return

    await websocket.send_text(json.dumps({
        "type": "comparison",
        "simulation_a": {
            "id": rec_a['id'],
            "timestamp": rec_a['timestamp'],
            "algorithm": rec_a['algorithm'],
            "summary": rec_a.get('summary', {}),
            "drones": rec_a.get('drones', [])
        },
        "simulation_b": {
            "id": rec_b['id'],
            "timestamp": rec_b['timestamp'],
            "algorithm": rec_b['algorithm'],
            "summary": rec_b.get('summary', {}),
            "drones": rec_b.get('drones', [])
        }
    }))
    await ws_log(websocket, f"Comparison: {id_a} vs {id_b}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
