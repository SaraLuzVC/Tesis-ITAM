# utils.py

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle
from adjustText import adjust_text


def plot_points(coords, names):
    """
    Plot labeled points on a 2D scatter plot.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(coords[:, 0], coords[:, 1], color="blue", s=50)

    texts = []
    for i, (x, y) in enumerate(coords):
        texts.append(ax.text(x, y, names[i], fontsize=8))

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Pueblos: solo puntos")
    ax.axis("equal")
    ax.grid(True)

    adjust_text(
        texts,
        ax=ax,
        only_move={"points": "y", "texts": "xy"},
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, shrinkA=10),
        expand_points=(1.2, 1.2),
        force_text=0.5,
        force_points=0.2,
    )
    plt.savefig('my_plot.jpg')
    plt.show()


def plot_beta_tangent_circles(coords, names, i, j, beta=1.0):
    """
    Plot two β-tangent circles between points i and j and highlight which points fall inside the lune.
    """
    p1, p2 = coords[i], coords[j]
    d = np.linalg.norm(p2 - p1)

    if beta < 1:
        raise ValueError("Beta debe ser ≥ 1 para una construcción válida.")

    r = (beta * d) / 2
    direction = (p2 - p1) / d
    center1 = p1 + direction * r
    center2 = p2 - direction * r

    fig, ax = plt.subplots(figsize=(10, 10))

    in_lune, out_lune = [], []
    for k, pk in enumerate(coords):
        if k in (i, j):
            continue
        if np.linalg.norm(pk - center1) < r or np.linalg.norm(pk - center2) < r:
            in_lune.append((pk, names[k]))
        else:
            out_lune.append((pk, names[k]))

    if out_lune:
        x_out, y_out = zip(*[p for p, _ in out_lune])
        ax.scatter(
            x_out, y_out, color="lightgray", label="Otros pueblos fuera de la luna"
        )
    if in_lune:
        x_in, y_in = zip(*[p for p, _ in in_lune])
        ax.scatter(x_in, y_in, color="blue", label="Pueblos dentro de la luna")

    ax.scatter(
        [p1[0], p2[0]], [p1[1], p2[1]], color="red", label="Puntos evaluados", zorder=3
    )
    ax.text(p1[0] + 0.0003, p1[1] + 0.0003, names[i], fontsize=9)
    ax.text(p2[0] + 0.0003, p2[1] + 0.0003, names[j], fontsize=9)

    ax.add_patch(
        Circle(
            center1,
            r,
            color="blue",
            fill=False,
            linestyle="--",
            label=f"Círculo tangente a {names[i]}",
        )
    )
    ax.add_patch(
        Circle(
            center2,
            r,
            color="green",
            fill=False,
            linestyle="--",
            label=f"Círculo tangente a {names[j]}",
        )
    )
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", alpha=0.5)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Círculos β = {beta} tangentes a '{names[i]}' y '{names[j]}'")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.savefig('my_plot.jpg')
    plt.show()


def plot_beta_lunes_for_node(coords, names, i, beta=1.0):
    """
    Plot β-lunes between a reference point i and all others in the dataset.
    """
    n = len(coords)
    p1 = coords[i]

    for j in range(n):
        if j == i:
            continue

        p2 = coords[j]
        d = np.linalg.norm(p2 - p1)
        if d == 0:
            continue

        r = (beta * d) / 2
        direction = (p2 - p1) / d
        center1 = p1 + direction * r
        center2 = p2 - direction * r

        in_lune, out_lune = [], []
        for k, pk in enumerate(coords):
            if k in (i, j):
                continue
            if np.linalg.norm(pk - center1) < r or np.linalg.norm(pk - center2) < r:
                in_lune.append((pk, names[k]))
            else:
                out_lune.append((pk, names[k]))

        fig, ax = plt.subplots(figsize=(8, 8))
        if out_lune:
            x_out, y_out = zip(*[p for p, _ in out_lune])
            ax.scatter(
                x_out, y_out, color="lightgray", label="Otros pueblos fuera de la luna"
            )
        if in_lune:
            x_in, y_in = zip(*[p for p, _ in in_lune])
            ax.scatter(x_in, y_in, color="blue", label="Pueblos dentro de la luna")

        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="red", zorder=3)
        ax.text(p1[0] + 0.0003, p1[1] + 0.0003, names[i], fontsize=9)
        ax.text(p2[0] + 0.0003, p2[1] + 0.0003, names[j], fontsize=9)

        ax.add_patch(
            Circle(
                center1,
                r,
                color="blue",
                fill=False,
                linestyle="--",
                label=f"Tangente a {names[i]}",
            )
        )
        ax.add_patch(
            Circle(
                center2,
                r,
                color="green",
                fill=False,
                linestyle="--",
                label=f"Tangente a {names[j]}",
            )
        )
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", alpha=0.5)

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(f"Luna β = {beta} entre '{names[i]}' y '{names[j]}'")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        plt.savefig('my_plot.jpg')
        plt.show()


def check_lune_for_node(coords, i, beta=1.0):
    """
    Determine for each pair (i, j) if the lune between them is empty of other points.
    Returns a binary vector indicating if an edge should exist (1) or not (0).
    """
    n = len(coords)
    p1 = coords[i]
    result_vector = []

    for j in range(n):
        if j == i:
            result_vector.append(0)
            continue

        p2 = coords[j]
        d = np.linalg.norm(p2 - p1)
        if d == 0:
            result_vector.append(0)
            continue

        r = (beta * d) / 2
        direction = (p2 - p1) / d
        center1 = p1 + direction * r
        center2 = p2 - direction * r

        is_clear = 1
        for k in range(n):
            if k == i or k == j:
                continue
            pk = coords[k]
            if np.linalg.norm(pk - center1) < r and np.linalg.norm(pk - center2) < r:
                is_clear = 0
                break

        result_vector.append(is_clear)

    return result_vector


def beta_skeleton_adjacency_matrix(coords, beta=1.0):
    """
    Construct a symmetric adjacency matrix for the β-skeleton graph.
    """
    n = len(coords)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        matrix[i] = check_lune_for_node(coords, i, beta=beta)

    return np.minimum(matrix, matrix.T)  # Ensure symmetry


def plot_beta_skeleton_graph(coords, names, beta=1.0):
    """
    Visualize the β-skeleton graph based on the adjacency matrix.
    """
    adj_matrix = beta_skeleton_adjacency_matrix(coords, beta=beta)
    G = nx.Graph()

    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(x, y), label=names[i])

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, "pos")
    labels = nx.get_node_attributes(G, "label")

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_color="skyblue", with_labels=False, node_size=100)

    texts = [plt.text(x, y, labels[i], fontsize=8) for i, (x, y) in pos.items()]
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, shrinkA=5),
        expand_points=(1.2, 1.2),
        force_text=0.5,
        force_points=0.2,
    )

    plt.title(f"Gráfica β-esqueleto (β = {beta})")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig('my_plot.jpg')
    plt.show()

################################## Naive

def euclidean_distance_matrix_symmetric(coords):
    """
    Computes a symmetric Euclidean distance matrix (in degrees),
    only calculating the upper triangle to save computation.

    Parameters:
        coords (np.ndarray): Array of shape (n, 2), where each row is [lon, lat].

    Returns:
        np.ndarray: A (n x n) symmetric matrix of distances.
    """
    n = coords.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric

    return dist_matrix


def compute_rng_adjacency_matrix(D):
    """
    Compute the adjacency matrix of the Relative Neighborhood Graph (RNG),
    based on the distance matrix D.

    Parameters:
        D (np.ndarray): Symmetric (n x n) distance matrix.

    Returns:
        adj (np.ndarray): Symmetric (n x n) RNG adjacency matrix (0 or 1).
    """
    n = D.shape[0]
    adj = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            d_ij = D[i, j]
            # if i == 68 and j == 70:
            #     print("distancia", i, "-", j, d_ij)
            # if i == 68 and j == 71:
            #     print("distancia", i, "-", j, d_ij)
            keep_edge = True

            for k in range(n):
                if k != i and k != j:
                    d_max_k = max(D[i, k], D[j, k])
                    # if i == 68 and j == 70 and k == 71:
                    #     print("d_max", k, d_max_k)
                    # if i == 68 and j == 71 and k == 70:
                    #     print("d_max", k, d_max_k)
                    if d_ij > d_max_k:
                        keep_edge = False
                        # if i == 68 and j == 70 and k == 71:
                        #     print(keep_edge)
                        # if i == 68 and j == 71 and k == 70:
                        #     print(keep_edge)
                        break

            if keep_edge:
                adj[i, j] = 1
                adj[j, i] = 1  # symmetry
                # if i == 68 and j == 70:
                #     print(keep_edge)
                # if i == 68 and j == 71:
                #     print(keep_edge)

    return adj