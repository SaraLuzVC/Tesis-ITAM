# =======================
# k-β (β=2) por prueba de "luna" + plot
# =======================
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import Delaunay, delaunay_plot_2d
from scipy.spatial import Delaunay, distance_matrix
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
import networkx as nx
from sweepcircles import *

from adjustText import adjust_text

# ---------- Geometría de las lunas (igual que tenías) ----------
from typing import Tuple, Union
Point = Tuple[float, float]
Weight = Union[float, Tuple[float, float]]

def create_lunes(p1: Point, p2: Point, beta: float, weight: Weight = 0.0):
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d == 0:
        raise ValueError("p1 and p2 must be distinct points.")
    ux, uy = dx / d, dy / d
    if isinstance(weight, tuple):
        if len(weight) != 2:
            raise ValueError("weight tuple must be (w1, w2).")
        w1, w2 = float(weight[0]), float(weight[1])
    else:
        w1 = w2 = float(weight)
    r = 0.5 * beta * d - (w1 + w2) / 2.0
    c1 = (x1 + (r + w1) * ux, y1 + (r + w1) * uy)
    c2 = (x2 - (r + w2) * ux, y2 - (r + w2) * uy)
    return [(c1, r), (c2, r)]

# ---------- Delaunay -> aristas (i<j) ----------
def compute_delaunay(coords: np.ndarray) -> np.ndarray:
    tri = Delaunay(coords)
    T = tri.simplices  # (k,3)
    E = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E.sort(axis=1)
    return np.unique(E, axis=0)

# ---------- Vecinos en la malla de aristas ----------
def _neighbors_of(idx: int, edges: np.ndarray) -> np.ndarray:
    mask = (edges[:, 0] == idx) | (edges[:, 1] == idx)
    inc = edges[mask]
    if inc.size == 0:
        return np.array([], dtype=int)
    neigh = np.where(inc[:, 0] == idx, inc[:, 1], inc[:, 0])
    return np.unique(neigh)

# ---------- Construcción del grafo k-β por prueba de luna ----------
def build_kbeta_graph_by_sweep(
    coords: np.ndarray,
    edges: np.ndarray | None = None,
    *,
    beta: float = 2.0,
    weight: Weight = 0.0,
    tolerance: int = 0,
    points_scope: str = "n_or_j",
    include_selected_points: bool = False,
    return_details: bool = False,
):
    E = compute_delaunay(coords) if edges is None else edges
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    details = []

    for (i, j) in E:
        p_i = tuple(map(float, coords[i]))
        p_j = tuple(map(float, coords[j]))
        circles = create_lunes(p_i, p_j, beta=beta, weight=weight)

        neigh_i = _neighbors_of(i, E)
        neigh_j = _neighbors_of(j, E)
        if points_scope == "n_only":
            candidate_indices = neigh_i
        elif points_scope == "n_or_j":
            candidate_indices = np.unique(np.concatenate([neigh_i, neigh_j]))
        else:
            raise ValueError("points_scope must be 'n_only' or 'n_or_j'")

        if include_selected_points:
            point_indices = candidate_indices
        else:
            point_indices = np.array([u for u in candidate_indices if u not in (i, j)], dtype=int)

        points = [tuple(map(float, coords[u])) for u in point_indices.tolist()]

        if len(points) == 0:
            M = [[0, 0], [0, 0]]
            offdiag = 0
        else:
            # Usa tu propia implementación ya existente de sweep_circles
            M = sweep_circles(circles, points)  # ← tu función
            # CAMBIO mínimo: lista de listas → indexa con [0][1]
            offdiag = int(M[0][1])  # == M[1][0]

        keep = (offdiag <= tolerance)
        if keep:
            G.add_edge(int(i), int(j))

        if return_details:
            details.append({
                "i": int(i),
                "j": int(j),
                "point_indices": point_indices.tolist(),
                "M": M,
                "offdiag": offdiag,
                "kept": keep,
            })

    return (G, details) if return_details else G

# ---------- Plot (tu versión con adjustText) ----------
def plot_rng_graph(coords, names, rng_graph):
    for i, (x, y) in enumerate(coords):
        rng_graph.nodes[i]['pos'] = (x, y)
        rng_graph.nodes[i]['label'] = names[i]
    pos = nx.get_node_attributes(rng_graph, 'pos')
    labels = nx.get_node_attributes(rng_graph, 'label')

    plt.figure(figsize=(12, 12))
    nx.draw(
        rng_graph,
        pos,
        node_color='skyblue',
        with_labels=False,
        node_size=100,
        edge_color='black'
    )
    texts = []
    for i, (x, y) in pos.items():
        texts.append(plt.text(x, y, labels[i], fontsize=8))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, shrinkA=5),
        expand_points=(1.2, 1.2),
        force_text=0.5,
        force_points=0.2
    )
    plt.title("Relative Neighborhood Graph (RNG)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    from itertools import combinations


def build_kbeta_graph_complete(
    coords: np.ndarray,
    edges: np.ndarray | None = None,  # ignored; kept for API-compat
    *,
    beta: float = 2.0,
    weight: Weight = 0.0,
    tolerance: int = 0,
    points_scope: str = "all",  # for complete: "all" (default) or "n_or_j" kept for compatibility
    include_selected_points: bool = False,
    return_details: bool = False,
):
    """
    Build k-beta graph by checking every possible edge of the complete graph (i<j).
    This does the full prune using sweep_circles(circles, points).

    NOTE: O(n^2) edges, each edge checks up to (n-2) points -> may be slow for large n.
    """
    n = len(coords)
    # All pairs i<j
    idx_pairs = np.array(list(combinations(range(n), 2)), dtype=int)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    details = []

    all_indices = np.arange(n, dtype=int)

    for i, j in idx_pairs:
        p_i = tuple(map(float, coords[int(i)]))
        p_j = tuple(map(float, coords[int(j)]))
        circles = create_lunes(p_i, p_j, beta=beta, weight=weight)

        # Candidate points for testing the lune:
        # For the complete-graph approach, normally we test ALL other points (except i,j),
        # unless include_selected_points True (then include them as well, but that doesn't make sense
        # logically so we keep behavior consistent: include_selected_points -> include i,j too).
        if points_scope not in ("all", "n_or_j", "n_only"):
            raise ValueError(
                "points_scope must be one of 'all', 'n_or_j', 'n_only' for complete graph"
            )

        if points_scope == "all" or points_scope == "n_or_j":
            candidate_indices = all_indices
        elif points_scope == "n_only":
            # In complete-graph mode there are no Delaunay neighbors; fall back to all other points
            candidate_indices = all_indices

        if include_selected_points:
            point_indices = candidate_indices
        else:
            # exclude i and j
            point_indices = candidate_indices[candidate_indices != i]
            point_indices = point_indices[point_indices != j]

        points = [tuple(map(float, coords[u])) for u in point_indices.tolist()]

        if len(points) == 0:
            M = [[0, 0], [0, 0]]
            offdiag = 0
        else:
            M = sweep_circles(circles, points)
            offdiag = int(M[0][1])

        keep = offdiag <= tolerance
        if keep:
            G.add_edge(int(i), int(j))

        if return_details:
            details.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "point_indices": point_indices.tolist(),
                    "M": M,
                    "offdiag": offdiag,
                    "kept": keep,
                }
            )

    return (G, details) if return_details else G


def build_kbeta_graph_auto(
    coords: np.ndarray,
    edges: np.ndarray | None = None,
    *,
    beta: float = 2.0,
    weight: Weight = 0.0,
    tolerance: int = 0,
    points_scope: str = "n_or_j",
    include_selected_points: bool = False,
    return_details: bool = False,
):
    """
    Auto selector:
      - if beta >= 1.0 -> use Delaunay + pruning (build_kbeta_graph_by_sweep)
      - if 0 <= beta < 1.0 -> use complete-graph pruning (build_kbeta_graph_complete)

    Preserves signature of build_kbeta_graph_by_sweep.
    """
    if beta >= 1.0:
        # Use the Delaunay-based approach (your existing function)
        # pass params through; edges will be computed inside if None
        return build_kbeta_graph_by_sweep(
            coords,
            edges=edges,
            beta=beta,
            weight=weight,
            tolerance=tolerance,
            points_scope=points_scope,
            include_selected_points=include_selected_points,
            return_details=return_details,
        )
    else:
        # Use complete graph approach
        # The complete-graph function ignores `edges`, but keep param for API compatibility
        # Map points_scope: default of complete uses "all"
        complete_scope = "all"
        return build_kbeta_graph_complete(
            coords,
            edges=None,
            beta=beta,
            weight=weight,
            tolerance=tolerance,
            points_scope=complete_scope,
            include_selected_points=include_selected_points,
            return_details=return_details,
        )
