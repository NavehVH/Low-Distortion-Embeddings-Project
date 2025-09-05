from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple, Optional
import heapq
import random


# Using our own implementation so will be adjusted to the ones needed in the essay
class Graph:
    def __init__(self):
        self.adjacency_lists: Dict[Hashable, Dict[Hashable, Dict]] = {}

    def add_node(self, vertex: Hashable) -> None:
        if vertex not in self.adjacency_lists:
            self.adjacency_lists[vertex] = {}

    def add_nodes_from(self, vertex_list: Iterable[Hashable]) -> None:
        for vertex in vertex_list:
            self.add_node(vertex)

    def add_edge(self, vertex_u: Hashable, vertex_v: Hashable, **edge_attributes) -> None:
        if vertex_u == vertex_v:
            self.add_node(vertex_u)
            self.add_node(vertex_v)
            return
        self.add_node(vertex_u)
        self.add_node(vertex_v)
        self.adjacency_lists[vertex_u][vertex_v] = dict(edge_attributes)
        self.adjacency_lists[vertex_v][vertex_u] = dict(edge_attributes)

    def remove_edge(self, vertex_u: Hashable, vertex_v: Hashable) -> None:
        if vertex_v in self.adjacency_lists.get(vertex_u, {}):
            del self.adjacency_lists[vertex_u][vertex_v]
        if vertex_u in self.adjacency_lists.get(vertex_v, {}):
            del self.adjacency_lists[vertex_v][vertex_u]

    def remove_edges_from(self, edge_list: Iterable[Tuple[Hashable, Hashable]]) -> None:
        for vertex_u, vertex_v in edge_list:
            self.remove_edge(vertex_u, vertex_v)

    def remove_node(self, vertex: Hashable) -> None:
        if vertex in self.adjacency_lists:
            for neighbor_vertex in list(self.adjacency_lists[vertex].keys()):
                del self.adjacency_lists[neighbor_vertex][vertex]
            del self.adjacency_lists[vertex]

    def neighbors(self, vertex: Hashable):
        return self.adjacency_lists.get(vertex, {}).keys()

    def nodes(self) -> List[Hashable]:
        return list(self.adjacency_lists.keys())

    def edges(self, data: bool = False):
        seen_edges = set()
        for vertex_u in self.adjacency_lists:
            for vertex_v, edge_attributes in self.adjacency_lists[vertex_u].items():
                if (vertex_v, vertex_u) in seen_edges:
                    continue
                seen_edges.add((vertex_u, vertex_v))
                yield (vertex_u, vertex_v, edge_attributes) if data else (vertex_u, vertex_v)

    def number_of_nodes(self) -> int:
        return len(self.adjacency_lists)

    def number_of_edges(self) -> int:
        return sum(len(neighbor_dict) for neighbor_dict in self.adjacency_lists.values()) // 2

    def has_edge(self, vertex_u: Hashable, vertex_v: Hashable) -> bool:
        return vertex_v in self.adjacency_lists.get(vertex_u, {})

    def copy(self) -> "Graph":
        copied_graph = Graph()
        for vertex_u, vertex_v, edge_attributes in self.edges(data=True):
            copied_graph.add_edge(vertex_u, vertex_v, **edge_attributes)
        return copied_graph


def _dijkstra_single_source(graph: Graph, source_vertex: Hashable, weight_attribute: Optional[str]) -> Dict[Hashable, float]:
    distances: Dict[Hashable, float] = {source_vertex: 0.0}
    priority_queue: List[Tuple[float, Hashable]] = [(0.0, source_vertex)]
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor_vertex in graph.adjacency_lists.get(current_vertex, {}):
            edge_weight = graph.adjacency_lists[current_vertex][neighbor_vertex].get(weight_attribute, 1.0) if weight_attribute else 1.0
            new_distance = current_distance + float(edge_weight)
            if neighbor_vertex not in distances or new_distance < distances[neighbor_vertex]:
                distances[neighbor_vertex] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor_vertex))
    return distances


def dijkstra_all_pairs(graph: Graph, weight_attribute: Optional[str] = None) -> Dict[Hashable, Dict[Hashable, float]]:
    return {source_vertex: _dijkstra_single_source(graph, source_vertex, weight_attribute) for source_vertex in graph.nodes()}


def gnm_random_graph(num_vertices: int, num_edges: int, seed: Optional[int] = None) -> Graph:
    random_generator = random.Random(seed)
    graph = Graph()
    graph.add_nodes_from(range(num_vertices))

    possible_edges = [(vertex_i, vertex_j) for vertex_i in range(num_vertices) for vertex_j in range(vertex_i + 1, num_vertices)]
    if num_edges > len(possible_edges):
        num_edges = len(possible_edges)

    selected_edges = random_generator.sample(possible_edges, num_edges)
    for vertex_u, vertex_v in selected_edges:
        graph.add_edge(vertex_u, vertex_v)
    return graph
