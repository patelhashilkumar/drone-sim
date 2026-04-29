# Multi-Drone 3D Path Optimizer

A high-performance simulation platform for real-time multi-drone 3D path optimization. The system utilizes the RRT* (Rapidly-exploring Random Tree Star) algorithm for pathfinding and B-Spline curves for trajectory smoothing. It features an interactive Three.js dashboard with live WebSocket telemetry and unsupervised learning capabilities for flight-path analysis.

Live Demo: [drone-sim-901y.onrender.com](https://drone-sim-901y.onrender.com)

![Dashboard](dashboard.png)

## Core Features

### Path Planning and Optimization
* **3D RRT* Algorithm**: Implements asymptotically optimal path planning in continuous 3D space, ensuring efficient navigation around complex obstacles.
* **Energy-Aware Cost Model**: Optimizes flight paths based on total Euclidean distance, altitude changes, and simulated wind drag factors.
* **B-Spline Trajectory Smoothing**: Automatically converts raw RRT* waypoints into smooth, flyable, and aerodynamically sound trajectories.

### Unsupervised Learning Analysis
* **Flight-Path Clustering**: Uses k-Means and DBSCAN algorithms to identify common route patterns and anomalies across multiple simulation runs.
* **Manifold Learning**: Employs Principal Component Analysis (PCA) to reduce high-dimensional path data into 2D scatter plots for intuitive visualization of flight clusters.
* **Representative Centroids**: Automatically calculates the mean path (centroid) for each identified cluster to highlight dominant navigation strategies.

### Interactive Simulation Environment
* **Real-Time Visualization**: Powered by Three.js (r128), providing a full-screen 3D dashboard with fluid camera controls and high-fidelity quadcopter models.
* **Dynamic Environment Editor**: Allows users to modify building heights, place new obstacles, and generate randomized city layouts in real-time.
* **WebSocket Telemetry**: Provides a low-latency bidirectional stream for live path updates, fleet telemetry, and simulation logs.

## Technical Specifications

| Layer | Technology |
|---|---|
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **Logic/Mathematics** | NumPy, SciPy (B-Splines), Scikit-Learn (Clustering) |
| **Frontend** | HTML5, TailwindCSS, Three.js |
| **Communication** | WebSockets (Real-time), REST API |
| **Deployment** | Render (Web Service) |

## Getting Started

### Prerequisites
* Python 3.8 or higher
* Modern web browser with WebGL support

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/patelhashilkumar/drone-sim.git
   cd drone-sim
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the simulation server:
   ```bash
   python server.py
   ```

4. Access the dashboard:
   Open `http://localhost:8000` in your web browser. The server automatically redirects to the interactive 3D interface.

## Control Interface

| Action | Input |
|---|---|
| **Orbit / Rotate** | Left-click + Drag |
| **Pan** | Middle-click + Drag |
| **Zoom** | Scroll wheel |
| **Open Settings** | Hamburger menu (top-left) |
| **View Telemetry** | Logs panel (top-right) |

## Project Architecture

* **server.py**: Main FastAPI entry point handling WebSocket connections, REST endpoints, and simulation state.
* **planner.py**: Implementation of the RRT* algorithm and B-Spline trajectory smoothing logic.
* **clustering.py**: Unsupervised learning pipeline for path resampling, feature extraction, and clustering.
* **environment.py**: Spatial management system for 3D grid bounds, obstacle collision detection, and city generation.
* **index.html**: Unified frontend containing the Three.js rendering engine and UI components.

## License

This project is licensed under the MIT License.