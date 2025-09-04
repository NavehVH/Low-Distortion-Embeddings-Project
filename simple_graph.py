#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple, Optional
import heapq
import random


class Graph:
    """
    Minimal undirected *simple* graph with edge-attribute dicts.

    Representation (dict-of-dict-of-dict):
        self.adj[u][v] -> dict of edge attributes (e.g., {"w": 1.7, "key": (...)})

    Notes
    -----
    - Undirected: we mirror entries, so u in adj[v] iff v in adj[u].
    - Simple: no parallel edges, no self-loops (u!=v). A new add_edge(u,v)
      overwrites attributes for the unique edge {u,v}.
    - Attribute storage: edges can hold arbitrary key/value pairs; we commonly
      use:
        • 'w'   : weight used by Dijkstra (if provided)
        • 'key' : strict total-order tuple used by Baswana–Sen (Sec 1.4)
        • 'orig_weight': original input weight copied into the spanner
    - Intention: keep this tiny and dependency-free to support the spanner code.
    """

    def __init__(self):
        # adj maps each node -> (neighbor -> attr dict)
        self.adj: Dict[Hashable, Dict[Hashable, Dict]] = {}

    # ---- structure ops ----

    def add_node(self, v: Hashable) -> None:
        """Add v if not present. O(1)."""
        if v not in self.adj:
            self.adj[v] = {}

    def add_nodes_from(self, nodes: Iterable[Hashable]) -> None:
        """Add a batch of nodes. Ignores duplicates. O(len(nodes))."""
        for v in nodes:
            self.add_node(v)

    def add_edge(self, u: Hashable, v: Hashable, **attrs) -> None:
        """
        Add/overwrite the simple undirected edge {u,v} with attributes.
        Self-loops are ignored by design (to ensure "simple graph").

        Complexity: O(1) average.
        """
        if u == v:
            # keep the graph simple; Baswana–Sen assumes simple undirected graphs
            self.add_node(u); self.add_node(v)
            return
        self.add_node(u); self.add_node(v)
        # undirected: write both ways
        self.adj[u][v] = dict(attrs)
        self.adj[v][u] = dict(attrs)

    def remove_edge(self, u: Hashable, v: Hashable) -> None:
        """
        Remove {u,v} if present. No-op if missing. O(1) average.
        """
        if v in self.adj.get(u, {}):
            del self.adj[u][v]
        if u in self.adj.get(v, {}):
            del self.adj[v][u]

    def remove_edges_from(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> None:
        """Batch removal; used heavily by the spanner's residual pruning."""
        for u, v in edges:
            self.remove_edge(u, v)

    def remove_node(self, v: Hashable) -> None:
        """
        Remove node v and all incident edges. O(deg(v)).
        """
        if v in self.adj:
            for u in list(self.adj[v].keys()):
                del self.adj[u][v]
            del self.adj[v]

    # ---- queries ----

    def neighbors(self, v: Hashable):
        """
        Iterator of neighbors of v. O(1) to access, O(deg(v)) to iterate.
        """
        return self.adj.get(v, {}).keys()

    def nodes(self) -> List[Hashable]:
        """List of nodes (keys in adj). O(n)."""
        return list(self.adj.keys())

    def edges(self, data: bool = False):
        """
        Iterate each undirected edge once. If data=True, yield (u,v,attrs).
        Complexity: O(n + m). Internal 'seen' avoids double-yielding {u,v}.
        """
        seen = set()
        for u in self.adj:
            for v, attrs in self.adj[u].items():
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                yield (u, v, attrs) if data else (u, v)

    def number_of_nodes(self) -> int:
        """|V|"""
        return len(self.adj)

    def number_of_edges(self) -> int:
        """|E| (each undirected edge counted once)."""
        return sum(len(nbrs) for nbrs in self.adj.values()) // 2

    def has_edge(self, u: Hashable, v: Hashable) -> bool:
        """Check membership of {u,v}. O(1) average."""
        return v in self.adj.get(u, {})

    def copy(self) -> "Graph":
        """
        Shallow copy of the edges and their attributes.
        NOTE: this copies only nodes that are incident to at least one edge.
        If you need isolated nodes as well, add them manually.
        """
        G2 = Graph()
        for u, v, attrs in self.edges(data=True):
            G2.add_edge(u, v, **attrs)
        return G2


# ---------- shortest paths (Dijkstra) ----------

def _dijkstra_one(G: Graph, src: Hashable, weight: Optional[str]) -> Dict[Hashable, float]:
    """
    Single-source Dijkstra (non-negative weights).
    - If 'weight' is None, every edge has unit length 1.0.
    - If 'weight' is a key (e.g., 'w'), use G.adj[u][v][weight] if present,
      else default to 1.0 (so missing weights don't crash).
    Returns a dict 'dist' with distances from src to all reachable nodes.

    Complexity: O((n + m) log n) with binary heap.
    """
    dist: Dict[Hashable, float] = {src: 0.0}
    pq: List[Tuple[float, Hashable]] = [(0.0, src)]  # (distance, node)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            # Outdated entry
            continue
        for v in G.adj.get(u, {}):
            # Weight fallback: if attribute not present, treat as 1.0
            w = G.adj[u][v].get(weight, 1.0) if weight else 1.0
            nd = d + float(w)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def dijkstra_all_pairs(G: Graph, weight: Optional[str] = None) -> Dict[Hashable, Dict[Hashable, float]]:
    """
    All-pairs distances via repeated Dijkstra.
    Returns: dict s -> (dict v -> dist(s,v))
    Used by the spanner's 'check_spanner' to verify the stretch condition.

    Complexity: O(n * (n + m) log n) with a binary heap; fine for the graphs
    this educational project targets. For very large graphs, consider
    replacing with something more scalable or using partial checks.
    """
    return {s: _dijkstra_one(G, s, weight) for s in G.nodes()}


# ---------- small helpers to build graphs ----------

def gnm_random_graph(n: int, m: int, seed: Optional[int] = None) -> Graph:
    """
    Uniform G(n,m) simple undirected graph.
    - Nodes are 0..n-1
    - m distinct edges sampled uniformly from all n*(n-1)/2 possibilities
    - No self-loops / no parallel edges by construction

    Notes
    -----
    - This uses a complete list of possible pairs; it's simple and uniform,
      but can be memory-heavy if n is very large. Good for testing Baswana–Sen.
    - We do *not* assign weights here; the caller can add a 'w' attribute later.

    Parameters
    ----------
    n : int
        Number of nodes (>= 0)
    m : int
        Number of edges (will be clipped to max possible if too large)
    seed : Optional[int]
        RNG seed for reproducibility

    Returns
    -------
    G : Graph
        A simple undirected G(n,m) instance.
    """
    rng = random.Random(seed)
    G = Graph()
    G.add_nodes_from(range(n))

    # List all unordered pairs (i<j). Size: n*(n-1)/2.
    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if m > len(possible):
        m = len(possible)

    # Uniformly sample m distinct edges (no replacement).
    edges = rng.sample(possible, m)
    for u, v in edges:
        G.add_edge(u, v)
    return G
