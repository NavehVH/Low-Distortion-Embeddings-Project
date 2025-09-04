#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import random
from typing import Dict, Hashable, Iterable, Optional, Tuple, List

from simple_graph import Graph, dijkstra_all_pairs, gnm_random_graph

# --------------------------
# Baswana–Sen Spanner (Section 4)
# --------------------------
# High-level: build a sparse subgraph H of G that preserves all-pairs
# distances within a multiplicative (2k-1) factor.
#
# Algorithm structure (matches Section 4):
#   • Preliminaries (Sec 1.4): assume a strict total order on edges.
#     We enforce it by storing a (weight, lo_id, hi_id) "key" per edge.
#   • Phase 1 (k-1 rounds): randomized clustering + local edge additions
#       Step 1: sample cluster centers with prob p = n^{-1/k}
#       Step 2: for each vertex, find its lightest incident edge to each
#               neighboring cluster (using the total-order key)
#       Step 3(a/3b): add a small set of representative edges and remove
#               certain residual edges according to the paper’s rules
#       Step 4: delete all intra-cluster edges from the residual graph
#     (If a round attempts to add "too many edges", re-sample the round.
#      This is the Las Vegas flavor; expected O(1) resamples.)
#   • Phase 2: vertex–cluster joining — add for every vertex the lightest
#              edge toward each adjacent cluster boundary still present.


def baswana_sen_spanner(
    G: Graph,
    stretch: int,
    weight: Optional[str] = None,
    seed: Optional[int] = None,
    *,
    max_iter_size_factor: float = 2.0,
) -> Graph:
    """
    Builds a (2k-1)-spanner in expected O(k*m).

    Parameters
    ----------
    G : Graph
        Undirected simple graph (custom minimal Graph).
    stretch : int
        Desired t >= 1. Effective guarantee is t' = 2*floor((t+1)/2) - 1 <= t.
    weight : Optional[str]
        Name of the edge attribute used as distance (None => unweighted).
    seed : Optional[int]
        RNG seed for reproducibility.
    max_iter_size_factor : float
        Per-iteration cap factor used for Las Vegas resampling (Section 4
        uses a Markov-style argument to bound expected additions).

    Returns
    -------
    H : Graph
        A subgraph of G that is a (2k-1)-spanner.
    """
    if stretch < 1:
        raise ValueError("stretch must be >= 1")
    rng = random.Random(seed)

    n = G.number_of_nodes()
    H = Graph()
    H.add_nodes_from(G.nodes())
    if n <= 1:
        return H  # trivial

    # k from requested t; this gives t' = 2k-1
    k = max(1, (stretch + 1) // 2)

    # Residual graph R (paper’s E'): copy of G with a strict total-order key.
    # This key implements the "distinct weights" assumption (Sec 1.4).
    R = _make_residual_graph(G, weight_attr=weight)

    # Initial clustering: every node is its own cluster, center = itself.
    clustering: Dict[Hashable, Hashable] = {v: v for v in G.nodes()}

    # Sampling probability p = n^{-1/k} (Section 4) and a cap for "too many edges".
    # If edges_to_add exceeds round_cap we resample this round (expected O(1) repeats).
    p = n ** (-1.0 / k)
    round_cap = int(max_iter_size_factor * (n ** (1.0 + 1.0 / k)))

    # --------------------------
    # Phase 1: k-1 clustering rounds (Section 4)
    # --------------------------
    for _ in range(max(0, k - 1)):
        while True:
            # Step 1 (Sec 4): sample cluster centers independently with prob p.
            sampled_centers = _sample_centers(clustering, p, rng)

            edges_to_add: set[Tuple[Hashable, Hashable]] = set()
            edges_to_remove: set[Tuple[Hashable, Hashable]] = set()
            new_clustering: Dict[Hashable, Hashable] = {}

            # Process only vertices that still appear in the residual graph R.
            for v in list(R.nodes()):
                # If v's current center was sampled, v stays with that center this round.
                if clustering[v] in sampled_centers:
                    new_clustering[v] = clustering[v]
                    continue

                # Step 2 (Sec 4): for each neighbor cluster C of v, pick the lightest edge
                # from v into C, using the strict total-order key.
                best_neighbor, best_key = _lightest_neighbor_per_cluster(R, clustering, v)
                if not best_neighbor:
                    # Isolated in R — no edges to process this round.
                    continue

                # Neighbor clusters among the sampled centers (if any).
                sampled_adj = [c for c in best_key if c in sampled_centers]

                if not sampled_adj:
                    # Step 3(a) (Sec 4): No sampled adjacent cluster.
                    # Add ONE representative edge from v to EACH neighboring cluster
                    # (these form “witness” edges ensuring bounded detours later),
                    # then remove ALL residual edges incident to v (finish v for this round).
                    for c, u in best_neighbor.items():
                        edges_to_add.add(_as_undirected(v, u))
                    for u in list(R.neighbors(v)):
                        edges_to_remove.add(_as_undirected(v, u))

                else:
                    # Step 3(b) (Sec 4): There is at least one sampled adjacent cluster.
                    # Connect v to the sampled adjacent cluster with the minimum key,
                    # reassign v to that cluster’s center, and also add edges to any
                    # cluster whose best edge is STRICTLY lighter than the chosen one.
                    closest_center = min(sampled_adj, key=lambda c: best_key[c])
                    closest_u = best_neighbor[closest_center]
                    closest_key = best_key[closest_center]

                    # Add edge from v to the closest sampled cluster
                    edges_to_add.add(_as_undirected(v, closest_u))
                    new_clustering[v] = closest_center

                    # Also add "too-good-to-miss" edges: those strictly lighter than closest_key.
                    for c, kkey in best_key.items():
                        if kkey < closest_key:
                            edges_to_add.add(_as_undirected(v, best_neighbor[c]))

                    # Prune R: delete residual new_clusteringedges from v into clusters whose best key
                    # is <= the chosen key (they won't be needed for stretch later).
                    for u in list(R.neighbors(v)):
                        c_u = clustering[u]
                        if c_u in best_key and best_key[c_u] <= closest_key:
                            edges_to_remove.add(_as_undirected(v, u))

            # Las Vegas cap (Markov trick from Sec 4): if this round tried to add too many edges,
            # resample the round; the expected number of retries is O(1).
            if len(edges_to_add) <= round_cap:
                # Commit the successful round.
                _bulk_add_edges(H, R, edges_to_add, weight_attr=weight)
                R.remove_edges_from(edges_to_remove)

                # Finalize clustering for this round:
                # nodes whose centers were sampled keep that center unless reassigned above.
                for node, center in clustering.items():
                    if center in sampled_centers and node not in new_clustering:
                        new_clustering[node] = center
                clustering = new_clustering

                # Step 4 (Sec 4): delete all edges inside clusters from the residual graph.
                _drop_intracluster_edges(R, clustering)

                # Remove from R any vertex that no longer belongs to a surviving cluster.
                for v in list(R.nodes()):
                    if v not in clustering:
                        R.remove_node(v)
                break  # iteration accepted
            else:
                # Too many edges this attempt; re-sample centers and redo this round.
                continue

    # --------------------------
    # Phase 2: vertex–cluster joining (Section 4)
    # --------------------------
    # After k-1 rounds, for each remaining vertex v, add one lightest edge
    # to every neighboring cluster (using the same "best per cluster" rule).
    # This completes the (2k-1) stretch guarantee.
    for v in list(R.nodes()):
        best_neighbor, _ = _lightest_neighbor_per_cluster(R, clustering, v)
        for u in best_neighbor.values():
            _add_one_edge(H, R, v, u, weight_attr=weight)

    return H

# --------------------------
# Internals
# --------------------------

def _make_residual_graph(G: Graph, weight_attr: Optional[str]) -> Graph:
    """Preliminaries (Sec 1.4): Build residual graph R with strict total-order keys.
       - For every edge (u,v) in G, store:
           key := (weight, repr(lo), repr(hi))  # strict, deterministic order
         where lo,hi is the sorted endpoint pair by repr().
       - The 'key' is used whenever the algorithm needs "the lightest edge".
    """
    R = Graph()
    R.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = data.get(weight_attr, 1.0) if weight_attr else 1.0
        lo, hi = (u, v) if repr(u) <= repr(v) else (v, u)
        key = (float(w), repr(lo), repr(hi))
        R.add_edge(u, v, key=key, orig_weight=float(w))
    return R

def _sample_centers(clustering: Dict[Hashable, Hashable], p: float, rng: random.Random) -> set:
    """Step 1 (Sec 4): Each current cluster center is sampled independently with prob p."""
    centers = set(clustering.values())
    return {c for c in centers if rng.random() < p}

def _lightest_neighbor_per_cluster(
    R: Graph,
    clustering: Dict[Hashable, Hashable],
    v: Hashable,
) -> Tuple[Dict[Hashable, Hashable], Dict[Hashable, Tuple]]:
    """Step 2 (Sec 4): For vertex v, find the lightest incident edge to EACH adjacent cluster.
       Returns:
         best_neighbor[c] -> neighbor u in cluster c minimizing edge key
         best_key[c]      -> that minimal key (acts like a strictly-ordered weight)
    """
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
    """Step 4 (Sec 4): Remove every edge whose endpoints lie in the same cluster."""
    to_remove: List[Tuple[Hashable, Hashable]] = []
    for u, v in R.edges():
        if clustering.get(u) == clustering.get(v):
            to_remove.append((u, v))
    R.remove_edges_from(to_remove)

def _as_undirected(u: Hashable, v: Hashable) -> Tuple[Hashable, Hashable]:
    """Normalize an undirected edge as an ordered tuple (lo, hi) by repr()."""
    return (u, v) if repr(u) <= repr(v) else (v, u)

def _bulk_add_edges(
    H: Graph,
    R: Graph,
    edges: Iterable[Tuple[Hashable, Hashable]],
    *,
    weight_attr: Optional[str],
) -> None:
    """Add a batch of edges into H, copying the original (input) weight if given."""
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
    """Add a single edge into H (if not already present)."""
    if H.has_edge(u, v):
        return
    H.add_edge(u, v)
    if weight_attr:
        H.adj[u][v][weight_attr] = R.adj[u][v]["orig_weight"]
        H.adj[v][u][weight_attr] = R.adj[u][v]["orig_weight"]

# --------------------------
# Validation helpers
# --------------------------

def check_spanner(G: Graph, H: Graph, t: float, weight: Optional[str] = None, atol: float = 1e-9) -> None:
    """Definition check: assert dist_H(u,v) <= t * dist_G(u,v) for all connected pairs."""
    dG = dijkstra_all_pairs(G, weight)
    dH = dijkstra_all_pairs(H, weight)
    for u in G.nodes():
        for v in G.nodes():
            if u == v:
                continue
            if v not in dG[u]:
                # disconnected in original; ignore this pair
                continue
            assert v in dH[u], f"No path in spanner between {u} and {v}"
            assert dH[u][v] <= t * dG[u][v] + atol, (
                f"Stretch violated for ({u},{v}): dH={dH[u][v]} vs {t}*dG={dG[u][v]}"
            )

def is_subgraph_edges(G: Graph, H: Graph) -> bool:
    """Sanity: all edges in H must also be in G."""
    return all(G.has_edge(u, v) for u, v in H.edges())

# --------------------------
# CLI
# --------------------------

def build_graph_from_args(args: argparse.Namespace) -> Tuple[Graph, Optional[str]]:
    if args.graph == "example":
        G = Graph()
        # A small "ladder with diagonals" example (weighted).
        # Rails: weight 1.0; rungs: 0.9; diagonals: heavier (2.0..3.0).
        # The algorithm should keep rails/rungs and often prune many diagonals.
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
        # Uniform G(n,m); optional random weights in [1,10).
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
        # Read "u v [w]" per line (nodes treated as strings).
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

    # Effective t' derived from requested t.
    k = max(1, (args.stretch + 1) // 2)
    t_eff = 2 * k - 1

    # Verify stretch property (full all-pairs check).
    ok = True
    try:
        check_spanner(G, H, t=t_eff, weight=weight_attr)
    except AssertionError as e:
        ok = False
        print("Stretch check failed:", e)

    # Report basic stats + theory-scale for sanity.
    n_nodes = G.number_of_nodes()
    m_G = G.number_of_edges()
    m_H = H.number_of_edges()
    expected_scale = k * (n_nodes ** (1.0 + 1.0 / k))  # ~ O(k * n^(1+1/k))
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
