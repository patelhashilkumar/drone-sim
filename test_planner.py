import asyncio
from environment import Environment
from planner import HybridPlanner

async def run_test():
    env = Environment(50, 50)
    drones_config = [
        {'id': 0, 'name': 'ALPHA',   'start': (5, 5, 1),   'goal': (45, 45, 1)},
        {'id': 1, 'name': 'BRAVO',   'start': (5, 45, 1),  'goal': (45, 5, 1)},
        {'id': 2, 'name': 'CHARLIE', 'start': (25, 5, 1),  'goal': (25, 45, 1)},
    ]
    for d in drones_config:
        env.add_drone(d['id'], d['start'], d['goal'])
    env.generate_city_blocks()
    
    for d in drones_config:
        print(f"Planning for drone {d['id']}...")
        planner = HybridPlanner(environment=env, drone_id=d['id'])
        path, metrics = planner.plan(num_points=120)
        import json
        import numpy as np
        try:
            test_payload = {
                "id": d['id'],
                "best_path": np.array(path).tolist(),
                "fitness": metrics["fitness"],
                "energy": metrics["energy"],
                "collisions": metrics["collisions"]
            }
            json.dumps(test_payload)
            print("JSON serialization SUCCESS")
        except Exception as e:
            print(f"JSON serialization FAILED: {e}")
        print(f"Path length: {len(path)}")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(run_test())
