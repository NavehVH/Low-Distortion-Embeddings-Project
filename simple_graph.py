from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple, Optional
import heapq
import random


class Graph:
    def __init__(self):
        self.adj: Dict[Hashable, Dict[Hashable, Dict]] = {}

    def add_node(self, v: Hashable) -> None:
        if v not in self.adj:
            self.adj[v] = {}

    def add_nodes_from(self, nodes: Iterable[Hashable]) -> None:
        for v in nodes:
            self.add_node(v)

    def add_edge(self, u: Hashable, v: Hashable, **attrs) -> None:
        if u == v:
            self.add_node(u);
            self.add_node(v)
            return
        self.add_node(u);
        self.add_node(v)
        self.adj[u][v] = dict(attrs)
        self.adj[v][u] = dict(attrs)

    def remove_edge(self, u: Hashable, v: Hashable) -> None:
        if v in self.adj.get(u, {}):
            del self.adj[u][v]
        if u in self.adj.get(v, {}):
            del self.adj[v][u]

    def remove_edges_from(self, edges: Iterable[Tuple[Hashable, Hashable]]) -> None:
        for u, v in edges:
            self.remove_edge(u, v)

    def remove_node(self, v: Hashable) -> None:
        if v in self.adj:
            for u in list(self.adj[v].keys()):
                del self.adj[u][v]
            del self.adj[v]

    def neighbors(self, v: Hashable):
        return self.adj.get(v, {}).keys()

    def nodes(self) -> List[Hashable]:
        return list(self.adj.keys())

    def edges(self, data: bool = False):
        seen = set()
        for u in self.adj:
            for v, attrs in self.adj[u].items():
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                yield (u, v, attrs) if data else (u, v)

    def number_of_nodes(self) -> int:
        return len(self.adj)

    def number_of_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self.adj.values()) // 2

    def has_edge(self, u: Hashable, v: Hashable) -> bool:
        return v in self.adj.get(u, {})

    def copy(self) -> "Graph":
        G2 = Graph()
        for u, v, attrs in self.edges(data=True):
            G2.add_edge(u, v, **attrs)
        return G2


def _dijkstra_one(G: Graph, src: Hashable, weight: Optional[str]) -> Dict[Hashable, float]:
    dist: Dict[Hashable, float] = {src: 0.0}
    pq: List[Tuple[float, Hashable]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in G.adj.get(u, {}):
            w = G.adj[u][v].get(weight, 1.0) if weight else 1.0
            nd = d + float(w)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def dijkstra_all_pairs(G: Graph, weight: Optional[str] = None) -> Dict[Hashable, Dict[Hashable, float]]:
    return {s: _dijkstra_one(G, s, weight) for s in G.nodes()}


def gnm_random_graph(n: int, m: int, seed: Optional[int] = None) -> Graph:
    rng = random.Random(seed)
    G = Graph()
    G.add_nodes_from(range(n))

    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if m > len(possible):
        m = len(possible)

    edges = rng.sample(possible, m)
    for u, v in edges:
        G.add_edge(u, v)
    return G
