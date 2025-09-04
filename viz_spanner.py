#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import argparse
from typing import Dict, Hashable, Optional, Tuple

# Your lightweight graph utils
from simple_graph import Graph, gnm_random_graph

# Matplotlib for drawing
import matplotlib.pyplot as plt


# ---------- load the algorithm from your file (spanner.py) ----------

def _load_algo():
    """
    Import baswana_sen_spanner from spanner.py (must be in the same folder).
    Raises a clear error if not found.
    """
    try:
        from spanner import baswana_sen_spanner
        return baswana_sen_spanner
    except Exception as e:
        raise RuntimeError(
            "Could not import baswana_sen_spanner from spanner.py. "
            "Ensure spanner.py is in the SAME folder and defines "
            "def baswana_sen_spanner(G, stretch, weight=None, seed=None, **kwargs)."
        ) from e


# ---------- layout & drawing ----------

def _circular_layout(nodes) -> Dict[Hashable, Tuple[float, float]]:
    """Deterministic circular layout (no NetworkX)."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    # Stable order for reproducible plots
    nodes.sort(key=lambda x: (str(type(x)), str(x)))
    pos: Dict[Hashable, Tuple[float, float]] = {}
    for i, v in enumerate(nodes):
        theta = 2.0 * math.pi * i / n
        pos[v] = (math.cos(theta), math.sin(theta))
    return pos


def _draw_graph(ax, G: Graph, pos: Dict[Hashable, Tuple[float, float]],
                title: str, weight_attr: Optional[str] = None) -> None:
    """Draw nodes, edges, and (optionally) weights."""
    # Edges
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1.2, zorder=1)

    # Nodes
    xs = [pos[v][0] for v in pos]
    ys = [pos[v][1] for v in pos]
    ax.scatter(xs, ys, s=160, zorder=3)

    # Node labels
    for v, (x, y) in pos.items():
        ax.text(
            x, y, str(v),
            ha="center", va="center", fontsize=9, color="white",
            bbox=dict(boxstyle="circle,pad=0.18", facecolor="black",
                      linewidth=0.0, alpha=0.6),
            zorder=4
        )

    # Optional edge-weight labels
    if weight_attr:
        for u, v in G.edges():
            if weight_attr in G.adj[u][v]:
                w = G.adj[u][v][weight_attr]
                mx = (pos[u][0] + pos[v][0]) / 2.0
                my = (pos[u][1] + pos[v][1]) / 2.0
                ax.text(mx, my, f"{float(w):.2f}", fontsize=7, alpha=0.75, zorder=5)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------- public API ----------

def visualize_original_and_spanner(G: Graph, H: Graph,
                                   weight_attr: Optional[str] = None,
                                   suptitle: Optional[str] = None,
                                   savepath: Optional[str] = None) -> None:
    """
    Side-by-side visualization of original graph G and spanner H.
    Shows |V|, |E|, and % edge reduction.
    """
    n = G.number_of_nodes()
    mG = G.number_of_edges()
    mH = H.number_of_edges()
    reduction = 0.0 if mG == 0 else 100.0 * (1.0 - mH / mG)

    # One shared layout so you can compare edges visually
    pos = _circular_layout(G.nodes())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _draw_graph(axes[0], G, pos,
                title=f"Original G  |  |V|={n}, |E|={mG}",
                weight_attr=weight_attr)
    _draw_graph(axes[1], H, pos,
                title=f"Spanner H  |  |V|={n}, |E|={mH}  |  ΔE={reduction:.2f}%",
                weight_attr=weight_attr)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    backend = plt.get_backend().lower()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight")
    elif "agg" in backend:  # headless
        auto_path = "spanner_viz.png"
        plt.savefig(auto_path, dpi=180, bbox_inches="tight")
        print(f"[headless] Saved figure to {auto_path}")
    else:
        plt.show()

    plt.close(fig)


def run_and_visualize(G: Graph, stretch: int = 3, weight_attr: Optional[str] = None,
                      seed: Optional[int] = None, savepath: Optional[str] = None) -> Graph:
    """
    Compute Baswana–Sen spanner using baswana_sen_spanner from spanner.py
    and visualize it next to the original graph.
    """
    baswana_sen_spanner = _load_algo()
    H = baswana_sen_spanner(G, stretch=stretch, weight=weight_attr, seed=seed)

    # For the title
    k = max(1, (stretch + 1) // 2)
    t_eff = 2 * k - 1

    visualize_original_and_spanner(
        G, H,
        weight_attr=weight_attr,
        suptitle=f"Baswana–Sen (2k-1)-spanner  |  k={k}  →  stretch={t_eff}",
        savepath=savepath,
    )
    return H


# ---------- CLI (demo runner) ----------

from typing import Tuple, Optional  # make sure these are imported

def _build_demo_graph(kind: str, n: int, m: int, weighted: bool,
                      seed: Optional[int], case: str) -> Tuple[Graph, Optional[str]]:
    if kind == "example":
        G = Graph()

        if case == "triangle":
            # 3 nodes, one heavy edge
            for a, b in [(0, 1), (0, 2)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(1, 2, w=2.0)
            return G, "w"

        elif case == "square":
            # C4 (unweighted cycle on 4 vertices)
            for a, b in [(0,1), (1,2), (2,3), (3,0)]:
                G.add_edge(a, b)  # unweighted
            return G, None

        elif case == "square_diag":
            # C4 with a diagonal (weighted)
            for a, b in [(0,1), (1,2), (2,3), (3,0)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(0, 2, w=2.0)   # heavier diagonal
            return G, "w"

        elif case == "path6":
            # P6 path on 6 vertices (unweighted)
            for a, b in [(0,1), (1,2), (2,3), (3,4), (4,5)]:
                G.add_edge(a, b)
            return G, None

        elif case == "star6":
            # Star with center 0 (unweighted)
            for leaf in [1,2,3,4,5]:
                G.add_edge(0, leaf)
            return G, None

        elif case == "k4_mix":
            # K4 with mixed weights
            # light triangle among {0,1,2}, heavier edges to 3
            edges = {
                (0,1):1.0, (0,2):1.0, (1,2):1.2,
                (0,3):2.0, (1,3):2.2, (2,3):2.4
            }
            for (u, v), w in edges.items():
                G.add_edge(u, v, w=w)
            return G, "w"

        elif case == "two_tris_bridge":
            # Two triangles connected by a light bridge
            for a, b in [(0,1), (1,2), (2,0)]:
                G.add_edge(a, b, w=1.0)
            for a, b in [(3,4), (4,5), (5,3)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(2, 3, w=0.5)  # bridge is the lightest edge
            return G, "w"
        
        elif case == "square_diag_equal":
            # Rectangle (0-1-2-3-0) with ONE diagonal (0-2); all edges weight 1.0
            for a, b in [(0,1), (1,2), (2,3), (3,0), (0,2)]:
                G.add_edge(a, b, w=1.0)
            return G, "w"
            
        elif case == "wheel6":
            # Center 0 connected to rim 1..5 (spokes are light), rim edges a bit heavier.
            # Expectation for k=2: if center 0 gets sampled, all rim vertices join it,
            # rim edges are pruned → spanner keeps ~ the 5 spokes (remove ~5 edges).
            for i in range(1, 6):
                G.add_edge(0, i, w=1.0)  # spokes (light)
            rim = [1, 2, 3, 4, 5]
            for i in range(len(rim)):
                u, v = rim[i], rim[(i + 1) % len(rim)]
                G.add_edge(u, v, w=1.8)  # rim (heavier)
            return G, "w"

        elif case == "ladder":
            # 2x4 ladder (rails light, rungs a bit heavier)
            # rails
            for a, b in [(0,1), (1,2), (2,3)]:
                G.add_edge(a, b, w=1.0)           # top rail
            for a, b in [(4,5), (5,6), (6,7)]:
                G.add_edge(a, b, w=1.0)           # bottom rail
            # rungs
            for a, b in [(0,4), (1,5), (2,6), (3,7)]:
                G.add_edge(a, b, w=1.8)
            return G, "w"

        elif case == "pentagon":
            # C5 (unweighted 5-cycle)
            for a, b in [(0,1), (1,2), (2,3), (3,4), (4,0)]:
                G.add_edge(a, b)
            return G, None

        else:
            raise SystemExit(f"Unknown --case: {case}")

    elif kind == "random":
        G = gnm_random_graph(n, m, seed=seed)
        if weighted:
            import random as _r
            rng = _r.Random(seed)
            for u, v in G.edges():
                w = 1.0 + rng.random() * 9.0
                G.adj[u][v]["w"] = w
                G.adj[v][u]["w"] = w
            return G, "w"
        return G, None

    else:
        raise SystemExit("kind must be 'example' or 'random'")



def main():
    parser = argparse.ArgumentParser(
        description="Visualize original graph vs. Baswana–Sen spanner (side-by-side)."
    )
    parser.add_argument(
    "--case",
    choices=[
        "triangle", "square", "square_diag", "path6", "star6",
        "k4_mix", "two_tris_bridge", "ladder", "pentagon",
        "wheel6", "k4_complete","square_diag_equal"    # ← NEW
    ],
    default="triangle",
    help="Which simple example graph to build when --graph example."
    )
    parser.add_argument("--graph", choices=["example", "random"], default="example")
    parser.add_argument("--n", type=int, default=20, help="random graph: number of nodes")
    parser.add_argument("--m", type=int, default=40, help="random graph: number of edges")
    parser.add_argument("--weighted", action="store_true", help="random graph: assign 'w' in [1,10)")
    parser.add_argument("--stretch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None, help="optional path to save the PNG")
    args = parser.parse_args()

    G, weight_attr = _build_demo_graph(args.graph, args.n, args.m, args.weighted, args.seed, args.case)
    run_and_visualize(G, stretch=args.stretch, weight_attr=weight_attr,
                      seed=args.seed, savepath=args.save)


if __name__ == "__main__":
    main()
