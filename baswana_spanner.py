from __future__ import annotations
import random
from typing import Dict, Hashable, Tuple, List
from simple_graph import Graph


def create_graph():
    graph = Graph()
    num_edges = int(input("Enter number of edges: "))
    print("Enter each edge as: vertex1 vertex2 weight (Example: 0 1 1.0)")
    for i in range(num_edges):
        vertex_u_str, vertex_v_str, weight_str = input(f"Edge {i + 1}: ").strip().split()
        graph.add_edge(int(vertex_u_str), int(vertex_v_str), w=float(weight_str))
    original_graph = graph
    k_parameter = max(1, (int(input("Enter stretch factor: ").strip()) + 1) // 2)
    random_seed = int(input("Enter random seed: ").strip())
    return original_graph, random_seed, k_parameter

def spanner_algorithm():
    spanner_graph = Graph()
    spanner_graph.add_nodes_from(original_graph.nodes())

    # Create residual graph with edge keys
    residual_graph = Graph()
    residual_graph.add_nodes_from(original_graph.nodes())
    for vertex_u, vertex_v, edge_data in original_graph.edges(data=True):
        edge_weight = edge_data.get("w", 1.0) if "w" else 1.0
        vertex_low, vertex_high = (vertex_u, vertex_v) if repr(vertex_u) <= repr(vertex_v) else (vertex_v, vertex_u)
        residual_graph.add_edge(vertex_u, vertex_v, key=(float(edge_weight), repr(vertex_low), repr(vertex_high)), orig_weight=float(edge_weight))

    # Phase 1: Forming clusters
    vertex_to_cluster_center = {vertex: vertex for vertex in original_graph.nodes()}
    sampling_probability = original_graph.number_of_nodes() ** (-1.0 / k_parameter)
    iteration_edge_capacity = int(2.0 * (original_graph.number_of_nodes() ** (1.0 + 1.0 / k_parameter)))
    for iteration_number in range(max(0, k_parameter - 1)):
        vertex_to_cluster_center = _execute_single_clustering_iteration(spanner_graph, residual_graph, vertex_to_cluster_center, sampling_probability, iteration_edge_capacity, random.Random(random_seed))

    # Phase 2: Vertex-vertex joining
    for vertex in list(residual_graph.nodes()):
        lightest_edge_to_cluster, _ = find_lightest_edges_to_clusters(residual_graph, vertex_to_cluster_center, vertex)
        for neighbor_vertex in lightest_edge_to_cluster.values():
            add_edge_to_spanner(spanner_graph, residual_graph, vertex, neighbor_vertex)

    return spanner_graph

def _execute_single_clustering_iteration(spanner_graph: Graph, residual_graph: Graph, current_clustering: Dict[Hashable, Hashable], sampling_probability: float, iteration_edge_capacity: int, random_generator: random.Random) -> Dict[Hashable, Hashable]:
    while True:
        sampled_cluster_centers = {center for center in set(current_clustering.values()) if random_generator.random() < sampling_probability}
        updated_clustering: Dict[Hashable, Hashable] = {}
        edges_to_add_to_spanner: set[Tuple[Hashable, Hashable]] = set()
        edges_to_remove_from_residual: set[Tuple[Hashable, Hashable]] = set()

        for vertex in list(residual_graph.nodes()):
            if current_clustering[vertex] in sampled_cluster_centers:
                updated_clustering[vertex] = current_clustering[vertex]
            else:
                lightest_edge_to_cluster, lightest_weight_to_cluster = find_lightest_edges_to_clusters(residual_graph, current_clustering, vertex)
                if lightest_edge_to_cluster:
                    adjacent_sampled_clusters = [cluster for cluster in lightest_weight_to_cluster if cluster in sampled_cluster_centers]
                    if not adjacent_sampled_clusters:
                        _process_isolated_vertex(vertex, residual_graph, lightest_edge_to_cluster, edges_to_add_to_spanner, edges_to_remove_from_residual)
                    else:
                        _process_connected_vertex(vertex, residual_graph, current_clustering, lightest_edge_to_cluster, lightest_weight_to_cluster, adjacent_sampled_clusters, updated_clustering, edges_to_add_to_spanner, edges_to_remove_from_residual)

        if len(edges_to_add_to_spanner) <= iteration_edge_capacity:
            return _commit_changes(current_clustering, edges_to_add_to_spanner, edges_to_remove_from_residual, residual_graph, sampled_cluster_centers, spanner_graph, updated_clustering)

def _process_isolated_vertex(vertex, residual_graph, lightest_edge_to_cluster, edges_to_add_to_spanner, edges_to_remove_from_residual):
    for cluster_center, neighbor_vertex in lightest_edge_to_cluster.items():
        edges_to_add_to_spanner.add(convert_to_undirected_edge(vertex, neighbor_vertex))

    for neighbor in list(residual_graph.neighbors(vertex)):
        edges_to_remove_from_residual.add(convert_to_undirected_edge(vertex, neighbor))

def _process_connected_vertex(vertex, residual_graph, current_clustering, lightest_edge_to_cluster, lightest_weight_to_cluster, adjacent_sampled_clusters, updated_clustering, edges_to_add_to_spanner, edges_to_remove_from_residual):
    closest_sampled_cluster = min(adjacent_sampled_clusters, key=lambda cluster: lightest_weight_to_cluster[cluster])
    edges_to_add_to_spanner.add(convert_to_undirected_edge(vertex, lightest_edge_to_cluster[closest_sampled_cluster]))
    updated_clustering[vertex] = closest_sampled_cluster

    for cluster_center, edge_weight in lightest_weight_to_cluster.items():
        if edge_weight < lightest_weight_to_cluster[closest_sampled_cluster]:
            edges_to_add_to_spanner.add(convert_to_undirected_edge(vertex, lightest_edge_to_cluster[cluster_center]))

    for neighbor in list(residual_graph.neighbors(vertex)):
        if current_clustering[neighbor] in lightest_weight_to_cluster and lightest_weight_to_cluster[
            current_clustering[neighbor]] <= lightest_weight_to_cluster[closest_sampled_cluster]:
            edges_to_remove_from_residual.add(convert_to_undirected_edge(vertex, neighbor))

def _commit_changes(current_clustering, edges_to_add_to_spanner, edges_to_remove_from_residual, residual_graph, sampled_cluster_centers, spanner_graph, updated_clustering):
    for vertex_u, vertex_v in edges_to_add_to_spanner:
        add_edge_to_spanner(spanner_graph, residual_graph, vertex_u, vertex_v)

    residual_graph.remove_edges_from(edges_to_remove_from_residual)

    for vertex, cluster_center in current_clustering.items():
        if cluster_center in sampled_cluster_centers and vertex not in updated_clustering:
            updated_clustering[vertex] = cluster_center
    current_clustering = updated_clustering

    # Remove intra-cluster edges
    edges_to_remove: List[Tuple[Hashable, Hashable]] = []
    for vertex_u, vertex_v in residual_graph.edges():
        if current_clustering.get(vertex_u) == current_clustering.get(vertex_v):
            edges_to_remove.append((vertex_u, vertex_v))
    residual_graph.remove_edges_from(edges_to_remove)

    for vertex in list(residual_graph.nodes()):
        if vertex not in current_clustering:
            residual_graph.remove_node(vertex)

    return current_clustering

def find_lightest_edges_to_clusters(residual_graph: Graph, vertex_to_cluster_center: Dict[Hashable, Hashable], source_vertex: Hashable, ) -> Tuple[Dict[Hashable, Hashable], Dict[Hashable, Tuple]]:
    lightest_edge_to_cluster: Dict[Hashable, Hashable] = {}
    lightest_weight_to_cluster: Dict[Hashable, Tuple] = {}

    for neighbor_vertex in residual_graph.neighbors(source_vertex):
        neighbor_cluster = vertex_to_cluster_center[neighbor_vertex]
        edge_key = residual_graph.adjacency_lists[source_vertex][neighbor_vertex]["key"]

        if (neighbor_cluster not in lightest_weight_to_cluster) or (edge_key < lightest_weight_to_cluster[neighbor_cluster]):
            lightest_weight_to_cluster[neighbor_cluster] = edge_key
            lightest_edge_to_cluster[neighbor_cluster] = neighbor_vertex

    return lightest_edge_to_cluster, lightest_weight_to_cluster

def add_edge_to_spanner(spanner_graph: Graph, residual_graph: Graph, vertex_u: Hashable, vertex_v: Hashable):
    if not spanner_graph.has_edge(vertex_u, vertex_v):
        spanner_graph.add_edge(vertex_u, vertex_v)
        original_weight = residual_graph.adjacency_lists[vertex_u][vertex_v]["orig_weight"]
        spanner_graph.adjacency_lists[vertex_u][vertex_v]["w"] = original_weight
        spanner_graph.adjacency_lists[vertex_v][vertex_u]["w"] = original_weight

def convert_to_undirected_edge(vertex_u: Hashable, vertex_v: Hashable) -> Tuple[Hashable, Hashable]:
    return (vertex_u, vertex_v) if repr(vertex_u) <= repr(vertex_v) else (vertex_v, vertex_u)


# -------- adapter for viz_spanner.py --------
def baswana_spanner(G, stretch=3, weight=None, seed=None, **kwargs):
    global original_graph, random_seed, k_parameter
    original_graph = G
    random_seed = 0 if seed is None else int(seed)
    k_parameter = max(1, (int(stretch) + 1) // 2)

    H = spanner_algorithm()
    return H
# -------- end adapter --------


if __name__ == "__main__":
    repeat = True
    while repeat:
        original_graph, random_seed, k_parameter = create_graph()

        while repeat:
            spanner_graph = spanner_algorithm()
            print(f"Original edges: {[(u, v, data.get('w', 1.0)) for u, v, data in original_graph.edges(data=True)]}")
            print(f"Spanner edges: {[(u, v, data.get('w', 1.0)) for u, v, data in spanner_graph.edges(data=True)]}")
            print(f"k = {k_parameter} | Edge reduction: {100.0 * (1.0 - spanner_graph.number_of_edges() / max(1, original_graph.number_of_edges())):.2f}%")
            repeat = (input("Run again with different seed? (y/n): ").strip().lower() == 'y')
            random_seed += 1

        repeat = (input("Run again with different parameters? (y/n): ").strip().lower() == 'y')
        

