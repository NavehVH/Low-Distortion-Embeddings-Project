from __future__ import annotations

import math
import argparse
import os
import random
from typing import Dict, Hashable, Optional, Tuple, List

from simple_graph import Graph, gnm_random_graph
import matplotlib.pyplot as plt

def _load_algo():
    try:
        from baswana_spanner import visual_spanner
        return visual_spanner
    except Exception as e:
        raise RuntimeError(
            "Could not import baswana_spanner from baswana_spanner.py. "
            "Ensure baswana_spanner.py defines "
            "def baswana_spanner(G, stretch, weight=None, seed=None, **kwargs)."
        ) from e


def _circular_layout(nodes) -> Dict[Hashable, Tuple[float, float]]:
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    nodes.sort(key=lambda x: (str(type(x)), str(x)))
    pos: Dict[Hashable, Tuple[float, float]] = {}
    for i, v in enumerate(nodes):
        theta = 2.0 * math.pi * i / n
        pos[v] = (math.cos(theta), math.sin(theta))
    return pos


def _draw_graph(ax, G, pos, *, weight_attr=None, title=None, edge_color="0.2"):
    xs = [pos[u][0] for u in G.nodes()]
    ys = [pos[u][1] for u in G.nodes()]
    ax.scatter(xs, ys, s=200, zorder=3)

    for u in G.nodes():
        ax.text(pos[u][0], pos[u][1] + 0.02, str(u),
                ha="center", va="bottom", fontsize=10)

    for (u, v, attrs) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], linewidth=2, color=edge_color, zorder=2)

        if weight_attr:
            w = attrs.get(weight_attr, None)
            if w is not None:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(w), fontsize=9)

    ax.set_title(title or "")
    ax.set_axis_off()

def visualize_original_and_spanner(G: Graph, H: Graph,
                                   weight_attr: Optional[str] = None,
                                   suptitle: Optional[str] = None,
                                   savepath: Optional[str] = None) -> None:
    n = G.number_of_nodes()
    mG = G.number_of_edges()
    mH = H.number_of_edges()
    reduction = 0.0 if mG == 0 else 100.0 * (1.0 - mH / mG)

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

    target_dir = os.path.dirname(savepath) if savepath else "outputs"
    if target_dir == "":
        target_dir = "outputs"
    os.makedirs(target_dir, exist_ok=True)

    if savepath:
        out = savepath
    else:
        out = os.path.join(target_dir, "spanner_viz.png")

    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


def run_and_visualize(G: Graph, stretch: int = 3,
                      weight_attr: Optional[str] = None,
                      seed: Optional[int] = None,
                      savepath: Optional[str] = None) -> Graph:
    baswana_sen_spanner = _load_algo()
    H = baswana_sen_spanner(G, stretch=stretch,
                            weight=weight_attr, seed=seed)

    k = max(1, (stretch + 1) // 2)
    t_eff = 2 * k - 1

    visualize_original_and_spanner(
        G, H,
        weight_attr=weight_attr,
        suptitle=f"Baswana–Sen (2k-1)-spanner  |  k={k}  →  stretch={t_eff}",
        savepath=savepath,
    )
    return H

def build_graph_from_edges(edges: List[Tuple[int, int, float]]) -> Tuple[Graph, str]:
    G = Graph()
    for (u, v, w) in edges:
        G.add_edge(u, v, w=float(w))
    return G, "w"


def _build_demo_graph(kind: str, n: int, m: int, weighted: bool,
                      seed: Optional[int], case: str) -> Tuple[Graph, Optional[str]]:
    if kind == "example":
        G = Graph()
        if case == "triangle":
            for a, b in [(0, 1), (0, 2)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(1, 2, w=2.0)
            return G, "w"
        elif case == "rectangle_diag":
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(0, 2, w=1.0)
            return G, "w"
        elif case == "rectangle_diag_alt":
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                G.add_edge(a, b, w=1.0)
            G.add_edge(1, 3, w=1.0)
            return G, "w"
        else:
            raise SystemExit(f"Unknown --case: {case}")
    elif kind == "random":
        G = gnm_random_graph(n, m, seed=seed)
        if weighted:
            rng = random.Random(seed)
            for u, v in G.edges():
                w = 1.0 + rng.random() * 9.0
                G.adjacency_lists[u][v]["w"] = w
                G.adjacency_lists[v][u]["w"] = w
            return G, "w"
        return G, None
    else:
        raise SystemExit("kind must be 'example' or 'random'")

TEST_SPECS = {
    "sanity_triangle": {
        "edges": [
            (0, 1, 1.0), (0, 2, 1.0), (1, 2, 2.0),
        ],
        "stretches": [1, 3],
    },
    "square_with_diagonal": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.0),
        ],
        "stretches": [1, 3],
    },
    "small_chain_extra": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
            (0, 4, 5.0), (1, 3, 2.5), (0, 2, 3.0),
        ],
        "stretches": [1, 3, 5],
    },
    "seed_variability": {
        "edges": [
            (0, 1, 2.5), (1, 2, 1.0), (2, 3, 3.0), (0, 3, 4.0),
        ],
        "stretches": [3],
    },
    "larger_redundancies": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
            (4, 5, 1.0), (0, 5, 7.0), (1, 3, 2.5), (2, 4, 2.5),
        ],
        "stretches": [3],
    },
    "triangle_plus_one": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0), (2, 3, 1.0),
        ],
        "stretches": [3],
    },
    "two_components": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0),
            (3, 4, 1.0), (4, 5, 1.0), (3, 5, 2.0),
        ],
        "stretches": [3],
    },
    "dense_boundish": {
        "edges": [
            (0, 1, 1.0), (0, 2, 1.5), (0, 3, 2.0), (0, 4, 2.5),
            (1, 2, 1.0), (1, 3, 1.5), (1, 4, 2.0),
            (2, 3, 1.0), (2, 4, 1.5),
            (3, 4, 1.0),
            (0, 5, 3.0), (1, 5, 2.5), (2, 5, 2.0), (3, 5, 1.5), (4, 5, 1.0),
        ],
        "stretches": [3],
    },
    "uniform_weights": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
            (0, 2, 1.0), (1, 3, 1.0),
        ],
        "stretches": [3],
    },
    "random_weights_same_topology": {
        "edges": [
            (0, 1, 2.3), (1, 2, 0.8), (2, 3, 4.1), (3, 0, 1.7),
            (0, 2, 3.5), (1, 3, 0.9),
        ],
        "stretches": [3],
    },
    "progressive_stretch": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 3.0),
            (0, 2, 2.5), (1, 3, 2.5), (0, 4, 1.0), (4, 2, 1.5),
        ],
        "stretches": [1, 3, 5, 7],
    },
    "small_scale_5": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
            (0, 4, 4.0), (1, 4, 3.0), (0, 2, 2.5),
        ],
        "stretches": [3],
    },
    "medium_scale_8": {
        "edges": [
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
            (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 0, 7.0),
            (0, 3, 3.5), (1, 4, 3.0), (2, 5, 3.0), (3, 6, 3.0),
        ],
        "stretches": [3],
    },
    "star_center": {
        "edges": [
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0),
        ],
        "stretches": [3],
    },
    "k4_complete": {
        "edges": [
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
            (1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0),
        ],
        "stretches": [3],
    },
}


def run_suite(suite: str, num_seeds: int = 3, fixed_seed: Optional[int] = None):
    if suite == "all":
        names = list(TEST_SPECS.keys())
    else:
        if suite not in TEST_SPECS:
            raise SystemExit(f"Unknown suite '{suite}'. "
                             f"Available: {', '.join(TEST_SPECS.keys())}, or 'all'")
        names = [suite]

    rng = random.Random()

    for name in names:
        spec = TEST_SPECS[name]
        edges = spec["edges"]
        stretches = spec["stretches"]

        if fixed_seed is not None:
            seeds = [fixed_seed]
        else:
            seeds = [rng.randint(0, 100) for _ in range(num_seeds)]

        G, w = build_graph_from_edges(edges)
        base_dir = os.path.join("outputs", "tests", name)
        os.makedirs(base_dir, exist_ok=True)

        for s in stretches:
            for sd in seeds:
                filename = f"{name}_stretch{s}_seed{sd}.png"
                out_path = os.path.join(base_dir, filename)
                print(f"[run] {name}: stretch={s}, seed={sd} -> {out_path}")
                run_and_visualize(G, stretch=s, weight_attr=w, seed=sd, savepath=out_path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=False)

    # Single mode
    p_single = sub.add_parser("single")
    p_single.add_argument("--case", choices=["triangle", "rectangle_diag", "rectangle_diag_alt"], default="triangle")
    p_single.add_argument("--graph", choices=["example", "random"], default="example")
    p_single.add_argument("--n", type=int, default=20)
    p_single.add_argument("--m", type=int, default=40)
    p_single.add_argument("--weighted", action="store_true")
    p_single.add_argument("--stretch", type=int, default=3)
    p_single.add_argument("--seed", type=int, default=None)
    p_single.add_argument("--save", type=str, default=None)
    p_single.set_defaults(default_mode=True)

    # Suite mode
    p_suite = sub.add_parser("suite")
    p_suite.add_argument("--name", type=str, default="all")
    p_suite.add_argument("--num-seeds", type=int, default=3)
    p_suite.add_argument("--fixed-seed", type=int, default=None)

    args = parser.parse_args()

    if getattr(args, "default_mode", False) or args.mode is None:
        seed = args.seed if args.seed is not None else random.randint(0, 10**9)
        G, weight_attr = _build_demo_graph(args.graph, args.n, args.m,
                                           args.weighted, seed, args.case)
        if args.save:
            os.makedirs("outputs", exist_ok=True)
            out = os.path.join("outputs", args.save)
        else:
            os.makedirs("outputs", exist_ok=True)
            out = os.path.join("outputs", f"{args.graph}_{args.case}_stretch{args.stretch}_seed{seed}.png")
        print(f"[single] graph={args.graph} case={args.case} stretch={args.stretch} seed={seed}")
        run_and_visualize(G, stretch=args.stretch, weight_attr=weight_attr, seed=seed, savepath=out)
        return

    if args.mode == "suite":
        run_suite(args.name, num_seeds=args.num_seeds, fixed_seed=args.fixed_seed)
        return


if __name__ == "__main__":
    main()