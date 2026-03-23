# 🚁 Multi-Drone 3D Path Optimizer

A real-time multi-drone 3D path optimization simulation using **RRT\*** (Rapidly-exploring Random Tree Star) with **B-Spline** trajectory smoothing. Features an interactive Three.js dashboard with live WebSocket telemetry.

![Dashboard](dashboard.png)

## ✨ Features

- **RRT\* Algorithm in 3D** — Asymptotically optimal path planning in continuous space
- **Energy-Aware Cost Model** — Optimizes for distance, altitude gain, and wind drag
- **B-Spline Trajectory Smoothing** — Converts raw waypoints into flyable, aerodynamic paths
- **Interactive 3D Dashboard** — Full-screen Three.js visualizer with Blender-style camera controls
- **Real-Time WebSocket Streaming** — Live path updates and fleet telemetry
- **Detailed Quadcopter Models** — Animated drones with spinning propellers, movement tilt, and hover wobble
- **Environment Editor** — Click to modify buildings, place obstacles, randomize city maps
- **Simulation History** — Auto-save, replay, and compare past simulation runs
- **Scenario Management** — Export/import full environment configurations as JSON
- **Collapsible UI** — Hamburger menu for controls, floating logs overlay, centered results popup

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **Math** | NumPy, SciPy (B-Splines) |
| **Frontend** | HTML5, TailwindCSS, Three.js (r128) |
| **Communication** | WebSockets (real-time), REST API |
| **Hosting** | Render (free tier) |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Local Development

```bash
# Clone the repository
git clone https://github.com/patelhashilkumar/drone-sim.git
cd drone-sim

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
```

Open **http://localhost:8000/static/index.html** in your browser.

### Deploy to Render

1. Fork or push this repo to your GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Select **Free** tier → **Create Web Service**
5. Live at `https://drone-sim.onrender.com/static/index.html`

## 🎮 Controls

| Action | Input |
|---|---|
| **Orbit / Rotate** | Left-click + Drag |
| **Pan** | Middle-click + Drag |
| **Zoom** | Scroll wheel |
| **Open Controls** | ☰ Hamburger menu (top-left) |
| **View Logs** | Logs panel (top-right) |

## 📖 Usage

1. **Connect** — Dashboard auto-connects to backend (green status dot)
2. **Start** — Click ▶ Start to begin RRT* path planning
3. **Tune** — Open ☰ menu to adjust Heuristic Weight, Spline Degree, Smoothness
4. **Edit** — Toggle Edit Mode, click buildings to change heights
5. **Add Obstacles** — Click Add Obstacle, then click ground to place
6. **Randomize** — Generate a fresh 3D city layout
7. **History** — View, replay, and compare past simulations

## 📁 Project Structure

```
├── server.py        # FastAPI server, WebSocket handler, simulation storage
├── planner.py       # RRT* search + B-Spline trajectory smoothing
├── environment.py   # 3D grid, obstacle management, bounds checking
├── index.html       # Three.js dashboard + UI (single-file frontend)
├── test_planner.py  # Unit tests for planner
├── requirements.txt # Python dependencies
├── render.yaml      # Render deployment config
└── README.md
```

## 📄 License

MIT License