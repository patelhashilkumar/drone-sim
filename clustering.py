"""
=============================================================================
  UNSUPERVISED LEARNING — Flight-Path Clustering
=============================================================================
  Clusters drone flight paths across simulation runs to discover common
  route patterns using k-Means and DBSCAN.

  Pipeline:
    1. Resample every path to a fixed number of 3D waypoints.
    2. Flatten (x, y, z) coordinates into a feature vector per path.
    3. Optionally append aggregate features (distance, avg altitude, …).
    4. Run k-Means or DBSCAN on the feature matrix.
    5. Reduce to 2D via PCA for scatter-plot visualization.
    6. Compute centroid paths (representative path per cluster).

  Public API:
    cluster_paths(simulations, algorithm, **kwargs) → ClusteringResult
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
RESAMPLE_POINTS = 50          # fixed waypoint count for every path
FEATURE_DIM     = RESAMPLE_POINTS * 3   # 150-D after flattening (x,y,z)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PathRecord:
    """One drone path from one simulation run."""
    sim_id: str
    drone_id: int
    drone_name: str
    path: List[List[float]]          # [[x,y,z], …]
    distance: float = 0.0
    energy: float = 0.0


@dataclass
class ClusteringResult:
    """Returned by cluster_paths()."""
    algorithm: str                   # 'kmeans' or 'dbscan'
    n_clusters: int
    labels: List[int]                # one label per path (-1 = noise)
    silhouette: float                # silhouette score (-1 if not computable)
    inertia: Optional[float]         # SSE (k-Means only)
    noise_count: int                 # DBSCAN noise points count
    centroid_paths: Dict[int, List[List[float]]]   # cluster_id → [[x,y,z],…]
    pca_points: List[Dict[str, Any]]               # [{x,y,label,sim_id,drone_name},…]
    path_records: List[Dict[str, Any]]              # metadata per path
    cluster_sizes: Dict[int, int]                   # cluster_id → count


# ═══════════════════════════════════════════════════════════════════════════════
#  PATH RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
def _resample_path(path: List[List[float]], n: int = RESAMPLE_POINTS) -> np.ndarray:
    """
    Resample a variable-length 3D path to exactly *n* equidistant points
    using linear interpolation along cumulative arc length.

    Returns:
        np.ndarray of shape (n, 3)
    """
    pts = np.array(path, dtype=np.float64)
    if len(pts) < 2:
        # Degenerate: single-point path → replicate
        return np.tile(pts[0], (n, 1))

    # Cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-9:
        return np.tile(pts[0], (n, 1))

    # Target arc-length positions
    targets = np.linspace(0, total_len, n)
    resampled = np.zeros((n, 3))
    for i, t in enumerate(targets):
        idx = np.searchsorted(cum_len, t, side='right') - 1
        idx = min(idx, len(pts) - 2)
        local_len = cum_len[idx + 1] - cum_len[idx]
        if local_len < 1e-12:
            resampled[i] = pts[idx]
        else:
            alpha = (t - cum_len[idx]) / local_len
            resampled[i] = pts[idx] * (1 - alpha) + pts[idx + 1] * alpha

    return resampled


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_features(records: List[PathRecord]) -> np.ndarray:
    """
    Convert a list of PathRecords into a (N, FEATURE_DIM+4) feature matrix.

    Features per path:
        - Flattened resampled (x,y,z) coordinates  → 150 dims
        - Total Euclidean distance                  → 1 dim
        - Mean altitude                             → 1 dim
        - Bounding-box diagonal                     → 1 dim
        - Mean curvature (avg angle between segments) → 1 dim
    """
    features = []
    for rec in records:
        resampled = _resample_path(rec.path)

        # Flatten spatial coordinates
        flat = resampled.flatten()  # (150,)

        # Aggregate features
        diffs = np.diff(resampled, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        total_dist = float(np.sum(seg_lens))
        mean_alt = float(np.mean(resampled[:, 2]))

        bb_min = resampled.min(axis=0)
        bb_max = resampled.max(axis=0)
        bb_diag = float(np.linalg.norm(bb_max - bb_min))

        # Mean curvature (angle between consecutive segments)
        if len(diffs) >= 2:
            v1 = diffs[:-1]
            v2 = diffs[1:]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            valid = (n1 > 1e-10) & (n2 > 1e-10)
            if np.any(valid):
                dots = np.sum(v1[valid] * v2[valid], axis=1)
                cos_a = np.clip(dots / (n1[valid] * n2[valid]), -1, 1)
                mean_curv = float(np.mean(np.arccos(cos_a)))
            else:
                mean_curv = 0.0
        else:
            mean_curv = 0.0

        vec = np.concatenate([flat, [total_dist, mean_alt, bb_diag, mean_curv]])
        features.append(vec)

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  CENTROID PATH COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_centroid_paths(
    records: List[PathRecord],
    labels: np.ndarray,
    n_clusters: int
) -> Dict[int, List[List[float]]]:
    """
    Compute the mean resampled path for each cluster.
    Returns a dict  cluster_id → [[x,y,z], …]  (RESAMPLE_POINTS points).
    """
    centroid_paths: Dict[int, List[List[float]]] = {}

    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_resampled = []
        for idx in np.where(mask)[0]:
            cluster_resampled.append(_resample_path(records[idx].path))
        mean_path = np.mean(cluster_resampled, axis=0)
        centroid_paths[cid] = mean_path.tolist()

    return centroid_paths


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def cluster_paths(
    simulations: List[Dict],
    algorithm: str = 'kmeans',
    k: int = 3,
    eps: float = 5.0,
    min_samples: int = 2
) -> ClusteringResult:
    """
    Cluster drone flight paths from saved simulation records.

    Args:
        simulations: List of simulation dicts (from SimulationStore.list_all / .get).
        algorithm:   'kmeans' or 'dbscan'.
        k:           Number of clusters for k-Means.
        eps:         Neighbourhood radius for DBSCAN.
        min_samples: Minimum samples for DBSCAN core points.

    Returns:
        ClusteringResult with labels, metrics, centroid paths, PCA scatter points.
    """
    # ── 1. Collect all drone paths ────────────────────────────────────────
    records: List[PathRecord] = []
    for sim in simulations:
        sim_id = sim.get('id', '?')
        for drone in sim.get('drones', []):
            path = drone.get('path', [])
            if len(path) < 2:
                continue
            records.append(PathRecord(
                sim_id=sim_id,
                drone_id=drone.get('id', 0),
                drone_name=drone.get('name', '?'),
                path=path,
                distance=drone.get('metrics', {}).get('distance', 0),
                energy=drone.get('metrics', {}).get('energy', 0),
            ))

    if len(records) < 2:
        # Can't cluster fewer than 2 paths
        return ClusteringResult(
            algorithm=algorithm, n_clusters=0, labels=[], silhouette=-1,
            inertia=None, noise_count=0, centroid_paths={}, pca_points=[],
            path_records=[], cluster_sizes={}
        )

    # ── 2. Feature extraction + scaling ───────────────────────────────────
    X_raw = _extract_features(records)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── 3. Clustering ─────────────────────────────────────────────────────
    inertia = None
    noise_count = 0

    if algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)
        noise_count = int(np.sum(labels == -1))
    else:
        # Default: k-Means
        actual_k = min(k, len(records))
        model = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        n_clusters = actual_k
        inertia = float(model.inertia_)

    # ── 4. Silhouette score ───────────────────────────────────────────────
    sil = -1.0
    n_unique = len(set(labels) - {-1})
    non_noise = np.sum(labels != -1)
    if n_unique >= 2 and non_noise >= 2:
        # Only compute on non-noise points
        mask = labels != -1
        if np.sum(mask) > n_unique:
            sil = float(silhouette_score(X[mask], labels[mask]))

    # ── 5. Centroid paths ─────────────────────────────────────────────────
    centroid_paths = _compute_centroid_paths(records, labels, n_clusters)

    # ── 6. PCA → 2D for scatter plot ──────────────────────────────────────
    n_components = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_2d = pca.fit_transform(X)

    pca_points = []
    for i, rec in enumerate(records):
        pca_points.append({
            'x': float(X_2d[i, 0]) if n_components >= 1 else 0,
            'y': float(X_2d[i, 1]) if n_components >= 2 else 0,
            'label': int(labels[i]),
            'sim_id': rec.sim_id,
            'drone_name': rec.drone_name,
            'drone_id': rec.drone_id,
        })

    # ── 7. Metadata per path ─────────────────────────────────────────────
    path_meta = []
    for i, rec in enumerate(records):
        path_meta.append({
            'sim_id': rec.sim_id,
            'drone_id': rec.drone_id,
            'drone_name': rec.drone_name,
            'label': int(labels[i]),
            'distance': rec.distance,
            'energy': rec.energy,
        })

    # ── 8. Cluster sizes ──────────────────────────────────────────────────
    cluster_sizes: Dict[int, int] = {}
    for lbl in labels:
        lbl_int = int(lbl)
        cluster_sizes[lbl_int] = cluster_sizes.get(lbl_int, 0) + 1

    return ClusteringResult(
        algorithm=algorithm,
        n_clusters=n_clusters,
        labels=[int(l) for l in labels],
        silhouette=round(sil, 4),
        inertia=round(inertia, 2) if inertia is not None else None,
        noise_count=noise_count,
        centroid_paths=centroid_paths,
        pca_points=pca_points,
        path_records=path_meta,
        cluster_sizes=cluster_sizes,
    )
