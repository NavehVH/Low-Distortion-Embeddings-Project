from __future__ import annotations
import argparse
import random
from typing import Dict, Hashable, Iterable, Optional, Tuple, List

from simple_graph import Graph, dijkstra_all_pairs, gnm_random_graph


def baswana_sen_spanner(
        G: Graph,
        stretch: int,
        weight: Optional[str] = None,
        seed: Optional[int] = 1,
        *,
        max_iter_size_factor: float = 2.0,
) -> Graph:
    if stretch < 1:
        raise ValueError("stretch must be >= 1")
    rng = random.Random(seed)

    n = G.number_of_nodes()
    H = Graph()
    H.add_nodes_from(G.nodes())
    if n <= 1:
        return H

    k = max(1, (stretch + 1) // 2)
    R = _make_residual_graph(G, weight_attr=weight)

    clustering: Dict[Hashable, Hashable] = {v: v for v in G.nodes()}

    p = n ** (-1.0 / k)
    round_cap = int(max_iter_size_factor * (n ** (1.0 + 1.0 / k)))

    for _ in range(max(0, k - 1)):
        while True:
            sampled_centers = _sample_centers(clustering, p, rng)

            edges_to_add: set[Tuple[Hashable, Hashable]] = set()
            edges_to_remove: set[Tuple[Hashable, Hashable]] = set()
            new_clustering: Dict[Hashable, Hashable] = {}

            for v in list(R.nodes()):
                if clustering[v] in sampled_centers:
                    new_clustering[v] = clustering[v]
                    continue

                best_neighbor, best_key = _lightest_neighbor_per_cluster(R, clustering, v)
                if not best_neighbor:
                    continue

                sampled_adj = [c for c in best_key if c in sampled_centers]

                if not sampled_adj:
                    for c, u in best_neighbor.items():
                        edges_to_add.add(_as_undirected(v, u))
                    for u in list(R.neighbors(v)):
                        edges_to_remove.add(_as_undirected(v, u))

                else:
                    closest_center = min(sampled_adj, key=lambda c: best_key[c])
                    closest_u = best_neighbor[closest_center]
                    closest_key = best_key[closest_center]
                    edges_to_add.add(_as_undirected(v, closest_u))
                    new_clustering[v] = closest_center

                    for c, kkey in best_key.items():
                        if kkey < closest_key:
                            edges_to_add.add(_as_undirected(v, best_neighbor[c]))

                    for u in list(R.neighbors(v)):
                        c_u = clustering[u]
                        if c_u in best_key and best_key[c_u] <= closest_key:
                            edges_to_remove.add(_as_undirected(v, u))

            if len(edges_to_add) <= round_cap:
                _bulk_add_edges(H, R, edges_to_add, weight_attr=weight)
                R.remove_edges_from(edges_to_remove)

                for node, center in clustering.items():
                    if center in sampled_centers and node not in new_clustering:
                        new_clustering[node] = center
                clustering = new_clustering

                _drop_intracluster_edges(R, clustering)

                for v in list(R.nodes()):
                    if v not in clustering:
                        R.remove_node(v)
                break
            else:
                continue

    for v in list(R.nodes()):
        best_neighbor, _ = _lightest_neighbor_per_cluster(R, clustering, v)
        for u in best_neighbor.values():
            _add_one_edge(H, R, v, u, weight_attr=weight)

    return H


def _make_residual_graph(G: Graph, weight_attr: Optional[str]) -> Graph:
    R = Graph()
    R.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = data.get(weight_attr, 1.0) if weight_attr else 1.0
        lo, hi = (u, v) if repr(u) <= repr(v) else (v, u)
        key = (float(w), repr(lo), repr(hi))
        R.add_edge(u, v, key=key, orig_weight=float(w))
    return R


def _sample_centers(clustering: Dict[Hashable, Hashable], p: float, rng: random.Random) -> set:
    centers = set(clustering.values())
    return {c for c in centers if rng.random() < p}


def _lightest_neighbor_per_cluster(
        R: Graph,
        clustering: Dict[Hashable, Hashable],
        v: Hashable,
) -> Tuple[Dict[Hashable, Hashable], Dict[Hashable, Tuple]]:
    best_neighbor: Dict[Hashable, Hashable] = {}
    best_key: Dict[Hashable, Tuple] = {}
    for u in R.neighbors(v):
        c = clustering[u]
        key = R.adj[v][u]["key"]
        if (c not in best_key) or (key < best_key[c]):
            best_key[c] = key
            best_neighbor[c] = u
    return best_neighbor, best_key


def _drop_intracluster_edges(R: Graph, clustering: Dict[Hashable, Hashable]) -> None:
    to_remove: List[Tuple[Hashable, Hashable]] = []
    for u, v in R.edges():
        if clustering.get(u) == clustering.get(v):
            to_remove.append((u, v))
    R.remove_edges_from(to_remove)


def _as_undirected(u: Hashable, v: Hashable) -> Tuple[Hashable, Hashable]:
    return (u, v) if repr(u) <= repr(v) else (v, u)


def _bulk_add_edges(
        H: Graph,
        R: Graph,
        edges: Iterable[Tuple[Hashable, Hashable]],
        *,
        weight_attr: Optional[str],
) -> None:
    for u, v in edges:
        _add_one_edge(H, R, u, v, weight_attr=weight_attr)


def _add_one_edge(
        H: Graph,
        R: Graph,
        u: Hashable,
        v: Hashable,
        *,
        weight_attr: Optional[str],
) -> None:
    if H.has_edge(u, v):
        return
    H.add_edge(u, v)
    if weight_attr:
        H.adj[u][v][weight_attr] = R.adj[u][v]["orig_weight"]
        H.adj[v][u][weight_attr] = R.adj[u][v]["orig_weight"]


def check_spanner(G: Graph, H: Graph, t: float, weight: Optional[str] = None, atol: float = 1e-9) -> None:
    dG = dijkstra_all_pairs(G, weight)
    dH = dijkstra_all_pairs(H, weight)
    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue
            if v not in dG[u]:
                continue
            assert v in dH[u], f"No path in spanner between {u} and {v}"
            assert dH[u][v] <= t * dG[u][v] + atol, (
                f"Stretch violated for ({u},{v}): dH={dH[u][v]} vs {t}*dG={dG[u][v]}"
            )


def is_subgraph_edges(G: Graph, H: Graph) -> bool:
    return all(G.has_edge(u, v) for u, v in H.edges())


def build_graph_from_args(args: argparse.Namespace) -> Tuple[Graph, Optional[str]]:
    if args.graph == "example":
        G = Graph()
        G.add_edge(0, 1, w=1.0)
        G.add_edge(1, 2, w=1.0)
        G.add_edge(2, 3, w=1.0)
        G.add_edge(4, 5, w=1.0)
        G.add_edge(5, 6, w=1.0)
        G.add_edge(6, 7, w=1.0)
        G.add_edge(0, 4, w=0.9)
        G.add_edge(1, 5, w=0.9)
        G.add_edge(2, 6, w=0.9)
        G.add_edge(3, 7, w=0.9)
        G.add_edge(0, 5, w=2.0)
        G.add_edge(1, 4, w=2.2)
        G.add_edge(1, 6, w=2.4)
        G.add_edge(2, 5, w=2.6)
        G.add_edge(2, 7, w=2.8)
        G.add_edge(3, 6, w=3.0)
        return G, "w"

    if args.graph == "random":
        G = gnm_random_graph(args.n, args.m, seed=args.seed)
        if args.weighted:
            rng = random.Random(args.seed)
            for u, v in G.edges():
                w = 1.0 + rng.random() * 9.0
                G.adj[u][v]["w"] = w
                G.adj[v][u]["w"] = w
            return G, "w"
        else:
            return G, None

    if args.graph == "edgelist":
        if not args.edgelist:
            raise SystemExit("--edgelist path is required when --graph edgelist")
        G = Graph()
        weight_attr = None
        with open(args.edgelist, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 2:
                    u, v = parts
                    G.add_edge(u, v)
                else:
                    u, v, w = parts[0], parts[1], float(parts[2])
                    G.add_edge(u, v, w=w)
                    weight_attr = "w"
        return G, weight_attr

    raise SystemExit(f"Unknown graph mode: {args.graph}")


def main():
    ap = argparse.ArgumentParser(description="Baswana–Sen (2k-1) spanner (no external deps)")
    ap.add_argument("--stretch", type=int, default=3, help="desired t (builds (2k-1)-spanner with k=floor((t+1)/2))")
    ap.add_argument("--graph", choices=["example", "random", "edgelist"], default="example", help="graph source")
    ap.add_argument("--edgelist", type=str, default=None, help="path to edgelist (u v [w]) for --graph edgelist")
    ap.add_argument("--n", type=int, default=100, help="random graph: number of nodes")
    ap.add_argument("--m", type=int, default=300, help="random graph: number of edges")
    ap.add_argument("--weighted", action="store_true", help="random graph: assign random weights 'w' in [1,10)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--print-edges", action="store_true", help="print spanner edges")
    args = ap.parse_args()

    G, weight_attr = build_graph_from_args(args)
    H = baswana_sen_spanner(G, stretch=args.stretch, weight=weight_attr, seed=args.seed)

    k = max(1, (args.stretch + 1) // 2)
    t_eff = 2 * k - 1

    ok = True
    try:
        check_spanner(G, H, t=t_eff, weight=weight_attr)
    except AssertionError as e:
        ok = False
        print("Stretch check failed:", e)

    n_nodes = G.number_of_nodes()
    m_G = G.number_of_edges()
    m_H = H.number_of_edges()
    expected_scale = k * (n_nodes ** (1.0 + 1.0 / k))
    reduction_pct = 100.0 * (1.0 - m_H / max(1, m_G))

    print("=== Baswana–Sen Spanner ===")
    print(f"Nodes: {n_nodes}  |  Edges(G): {m_G}  |  Edges(H): {m_H}")
    print(f"Requested stretch: {args.stretch}  ->  Actual guarantee: {t_eff}")
    print(f"Weight attr: {weight_attr!r}")
    print(f"Subgraph of G: {is_subgraph_edges(G, H)}")
    print(f"Stretch check: {'OK' if ok else 'FAILED'}")
    print(f"Expected O-scale (~k*n^(1+1/k)): {expected_scale:.1f}")
    print(f"Edge reduction: {reduction_pct:.2f}%")

    if args.print_edges:
        print("Spanner edges (u, v, data):")
        for u, v, data in H.edges(data=True):
            print((u, v, data))


if __name__ == "__main__":
    main()
