from __future__ import annotations
import random
from typing import Dict, Hashable, Iterable, Optional, Tuple, List

from simple_graph import Graph

def baswana_sen_spanner(
        original_graph: Graph,
        stretch: int,
        weight_attribute: Optional[str] = None,
        random_seed: Optional[int] = 1,
        *,
        max_iteration_size_factor: float = 2.0,
) -> Graph:
    if stretch < 1:
        raise ValueError("stretch must be >= 1")
    
    random_generator = random.Random(random_seed)
    num_vertices = original_graph.number_of_nodes()

    spanner_graph = Graph()
    spanner_graph.add_nodes_from(original_graph.nodes())
    
    if num_vertices <= 1:
        return spanner_graph

    k_parameter = max(1, (stretch + 1) // 2)
    residual_graph = _create_residual_graph_with_edge_keys(original_graph, weight_attribute)
    vertex_to_cluster_center: Dict[Hashable, Hashable] = {vertex: vertex for vertex in original_graph.nodes()}
    
    sampling_probability = num_vertices ** (-1.0 / k_parameter)
    iteration_edge_capacity = int(max_iteration_size_factor * (num_vertices ** (1.0 + 1.0 / k_parameter)))
    
    vertex_to_cluster_center = _phase1_form_clusters(spanner_graph, residual_graph, vertex_to_cluster_center, k_parameter, sampling_probability, iteration_edge_capacity, random_generator, weight_attribute)
    _phase2_vertex_cluster_joining(spanner_graph, residual_graph, vertex_to_cluster_center, weight_attribute)
    
    return spanner_graph

def _phase1_form_clusters(
        spanner_graph: Graph,
        residual_graph: Graph,
        vertex_to_cluster_center: Dict[Hashable, Hashable],
        k_parameter: int,
        sampling_probability: float,
        iteration_edge_capacity: int,
        random_generator: random.Random,
        weight_attribute: Optional[str]
) -> Dict[Hashable, Hashable]:
    current_clustering = vertex_to_cluster_center
    for iteration_number in range(max(0, k_parameter - 1)):
        current_clustering = _execute_single_clustering_iteration(spanner_graph, residual_graph, current_clustering, sampling_probability, iteration_edge_capacity, random_generator, weight_attribute)
    return current_clustering

def _execute_single_clustering_iteration(
        spanner_graph: Graph,
        residual_graph: Graph,
        current_clustering: Dict[Hashable, Hashable],
        sampling_probability: float,
        iteration_edge_capacity: int,
        random_generator: random.Random,
        weight_attribute: Optional[str]
) -> Dict[Hashable, Hashable]:
    while True:
        sampled_cluster_centers = _step1_sample_cluster_centers(current_clustering, sampling_probability, random_generator)
        
        edges_to_add_to_spanner: set[Tuple[Hashable, Hashable]] = set()
        edges_to_remove_from_residual: set[Tuple[Hashable, Hashable]] = set()
        updated_clustering: Dict[Hashable, Hashable] = {}
        
        for vertex in list(residual_graph.nodes()):
            if current_clustering[vertex] in sampled_cluster_centers:
                updated_clustering[vertex] = current_clustering[vertex]
                continue
            
            lightest_edge_to_cluster, lightest_weight_to_cluster = _step2_find_lightest_edges_to_clusters(residual_graph, current_clustering, vertex)
            
            if not lightest_edge_to_cluster:
                continue
            
            adjacent_sampled_clusters = [cluster for cluster in lightest_weight_to_cluster if cluster in sampled_cluster_centers]
            
            if not adjacent_sampled_clusters:
                _handle_case_vertex_not_adjacent_to_sampled_clusters(vertex, residual_graph, lightest_edge_to_cluster, edges_to_add_to_spanner, edges_to_remove_from_residual)
            else:
                _handle_case_vertex_adjacent_to_sampled_clusters(vertex, residual_graph, current_clustering, lightest_edge_to_cluster, lightest_weight_to_cluster, adjacent_sampled_clusters, updated_clustering, edges_to_add_to_spanner, edges_to_remove_from_residual)
        
        if len(edges_to_add_to_spanner) <= iteration_edge_capacity:
            _add_edges_to_spanner(spanner_graph, residual_graph, edges_to_add_to_spanner, weight_attribute)
            residual_graph.remove_edges_from(edges_to_remove_from_residual)
            
            for vertex, cluster_center in current_clustering.items():
                if cluster_center in sampled_cluster_centers and vertex not in updated_clustering:
                    updated_clustering[vertex] = cluster_center

            current_clustering = updated_clustering
            _step4_remove_intra_cluster_edges(residual_graph, current_clustering)
            _remove_vertices_not_in_clustering(residual_graph, current_clustering)
            
            return current_clustering

def _step1_sample_cluster_centers(
        vertex_to_cluster_center: Dict[Hashable, Hashable], 
        sampling_probability: float, 
        random_generator: random.Random
) -> set:
    all_cluster_centers = set(vertex_to_cluster_center.values())
    return {center for center in all_cluster_centers if random_generator.random() < sampling_probability}

def _step2_find_lightest_edges_to_clusters(
        residual_graph: Graph,
        vertex_to_cluster_center: Dict[Hashable, Hashable],
        source_vertex: Hashable,
) -> Tuple[Dict[Hashable, Hashable], Dict[Hashable, Tuple]]:
    lightest_edge_to_cluster: Dict[Hashable, Hashable] = {}
    lightest_weight_to_cluster: Dict[Hashable, Tuple] = {}
    
    for neighbor_vertex in residual_graph.neighbors(source_vertex):
        neighbor_cluster = vertex_to_cluster_center[neighbor_vertex]
        edge_key = residual_graph.adjacency_lists[source_vertex][neighbor_vertex]["key"]
        
        if (neighbor_cluster not in lightest_weight_to_cluster) or (edge_key < lightest_weight_to_cluster[neighbor_cluster]):
            lightest_weight_to_cluster[neighbor_cluster] = edge_key
            lightest_edge_to_cluster[neighbor_cluster] = neighbor_vertex
    
    return lightest_edge_to_cluster, lightest_weight_to_cluster

def _handle_case_vertex_not_adjacent_to_sampled_clusters(
        vertex: Hashable,
        residual_graph: Graph,
        lightest_edge_to_cluster: Dict[Hashable, Hashable],
        edges_to_add_to_spanner: set[Tuple[Hashable, Hashable]],
        edges_to_remove_from_residual: set[Tuple[Hashable, Hashable]]
) -> None:
    for cluster_center, neighbor_vertex in lightest_edge_to_cluster.items():
        edges_to_add_to_spanner.add(_convert_to_undirected_edge(vertex, neighbor_vertex))
    
    for neighbor in list(residual_graph.neighbors(vertex)):
        edges_to_remove_from_residual.add(_convert_to_undirected_edge(vertex, neighbor))

def _handle_case_vertex_adjacent_to_sampled_clusters(
        vertex: Hashable,
        residual_graph: Graph,
        current_clustering: Dict[Hashable, Hashable],
        lightest_edge_to_cluster: Dict[Hashable, Hashable],
        lightest_weight_to_cluster: Dict[Hashable, Tuple],
        adjacent_sampled_clusters: List[Hashable],
        updated_clustering: Dict[Hashable, Hashable],
        edges_to_add_to_spanner: set[Tuple[Hashable, Hashable]],
        edges_to_remove_from_residual: set[Tuple[Hashable, Hashable]]
) -> None:
    closest_sampled_cluster = min(adjacent_sampled_clusters, key=lambda cluster: lightest_weight_to_cluster[cluster])
    closest_neighbor_vertex = lightest_edge_to_cluster[closest_sampled_cluster]
    closest_edge_weight = lightest_weight_to_cluster[closest_sampled_cluster]
    
    edges_to_add_to_spanner.add(_convert_to_undirected_edge(vertex, closest_neighbor_vertex))
    updated_clustering[vertex] = closest_sampled_cluster
    
    for cluster_center, edge_weight in lightest_weight_to_cluster.items():
        if edge_weight < closest_edge_weight:
            edges_to_add_to_spanner.add(_convert_to_undirected_edge(vertex, lightest_edge_to_cluster[cluster_center]))
    
    for neighbor in list(residual_graph.neighbors(vertex)):
        neighbor_cluster = current_clustering[neighbor]
        if (neighbor_cluster in lightest_weight_to_cluster and 
            lightest_weight_to_cluster[neighbor_cluster] <= closest_edge_weight):
            edges_to_remove_from_residual.add(_convert_to_undirected_edge(vertex, neighbor))

def _step4_remove_intra_cluster_edges(residual_graph: Graph, vertex_to_cluster_center: Dict[Hashable, Hashable]) -> None:
    edges_to_remove: List[Tuple[Hashable, Hashable]] = []
    for vertex_u, vertex_v in residual_graph.edges():
        if vertex_to_cluster_center.get(vertex_u) == vertex_to_cluster_center.get(vertex_v):
            edges_to_remove.append((vertex_u, vertex_v))
    residual_graph.remove_edges_from(edges_to_remove)

def _phase2_vertex_cluster_joining(
        spanner_graph: Graph,
        residual_graph: Graph,
        vertex_to_cluster_center: Dict[Hashable, Hashable],
        weight_attribute: Optional[str]
) -> None:
    for vertex in list(residual_graph.nodes()):
        lightest_edge_to_cluster, _ = _step2_find_lightest_edges_to_clusters(residual_graph, vertex_to_cluster_center, vertex)
        for neighbor_vertex in lightest_edge_to_cluster.values():
            _add_single_edge_to_spanner(spanner_graph, residual_graph, vertex, neighbor_vertex, weight_attribute)

def _create_residual_graph_with_edge_keys(original_graph: Graph, weight_attribute: Optional[str]) -> Graph:
    residual_graph = Graph()
    residual_graph.add_nodes_from(original_graph.nodes())
    
    for vertex_u, vertex_v, edge_data in original_graph.edges(data=True):
        edge_weight = edge_data.get(weight_attribute, 1.0) if weight_attribute else 1.0
        vertex_low, vertex_high = (vertex_u, vertex_v) if repr(vertex_u) <= repr(vertex_v) else (vertex_v, vertex_u)
        edge_key = (float(edge_weight), repr(vertex_low), repr(vertex_high))
        residual_graph.add_edge(vertex_u, vertex_v, key=edge_key, orig_weight=float(edge_weight))
    
    return residual_graph

def _remove_vertices_not_in_clustering(residual_graph: Graph, vertex_to_cluster_center: Dict[Hashable, Hashable]) -> None:
    for vertex in list(residual_graph.nodes()):
        if vertex not in vertex_to_cluster_center:
            residual_graph.remove_node(vertex)

def _convert_to_undirected_edge(vertex_u: Hashable, vertex_v: Hashable) -> Tuple[Hashable, Hashable]:
    return (vertex_u, vertex_v) if repr(vertex_u) <= repr(vertex_v) else (vertex_v, vertex_u)

def _add_edges_to_spanner(
        spanner_graph: Graph,
        residual_graph: Graph,
        edges: Iterable[Tuple[Hashable, Hashable]],
        weight_attribute: Optional[str],
) -> None:
    for vertex_u, vertex_v in edges:
        _add_single_edge_to_spanner(spanner_graph, residual_graph, vertex_u, vertex_v, weight_attribute)

def _add_single_edge_to_spanner(
        spanner_graph: Graph,
        residual_graph: Graph,
        vertex_u: Hashable,
        vertex_v: Hashable,
        weight_attribute: Optional[str],
) -> None:
    if spanner_graph.has_edge(vertex_u, vertex_v):
        return
    
    spanner_graph.add_edge(vertex_u, vertex_v)
    if weight_attribute:
        original_weight = residual_graph.adjacency_lists[vertex_u][vertex_v]["orig_weight"]
        spanner_graph.adjacency_lists[vertex_u][vertex_v][weight_attribute] = original_weight
        spanner_graph.adjacency_lists[vertex_v][vertex_u][weight_attribute] = original_weight


def main():
    graph = Graph()
    weight_attribute = "w"
    num_edges = int(input("Enter number of edges: "))
    print("Enter each edge as: vertex1 vertex2 weight (Example: 0 1 1.0)")
    
    for i in range(num_edges):
        edge_input = input(f"Edge {i + 1}: ").strip().split()
        vertex_u_str, vertex_v_str, weight_str = edge_input
        graph.add_edge(int(vertex_u_str), int(vertex_v_str), w=float(weight_str))
    
    original_graph = graph
    stretch_input = input("Enter stretch factor (default 3): ").strip()
    stretch = int(stretch_input) if stretch_input else 3
    
    seed_input = input("Enter random seed (default is None): ").strip()
    random_seed = int(seed_input) if seed_input else None
    
    spanner_graph = baswana_sen_spanner(original_graph, stretch=stretch, weight_attribute=weight_attribute, random_seed=random_seed)
    k_parameter = max(1, (stretch + 1) // 2)
    original_edges = original_graph.number_of_edges()
    spanner_edges = spanner_graph.number_of_edges()
    reduction_percentage = 100.0 * (1.0 - spanner_edges / max(1, original_edges))

    def format_edge(u, v, data):
        if weight_attribute and weight_attribute in data:
            return f"({u}, {v}, {data[weight_attribute]})"
        else:
            return f"({u}, {v})"

    original_edges_list = []
    for vertex_u, vertex_v, edge_data in original_graph.edges(data=True):
        original_edges_list.append(format_edge(vertex_u, vertex_v, edge_data))
    
    spanner_edges_list = []
    for vertex_u, vertex_v, edge_data in spanner_graph.edges(data=True):
        spanner_edges_list.append(format_edge(vertex_u, vertex_v, edge_data))
    
    print(f"\nOriginal: {', '.join(original_edges_list)}")
    print(f"Spanner: {', '.join(spanner_edges_list)} | k = {k_parameter} | Edge reduction: {reduction_percentage:.2f}%")

if __name__ == "__main__":
    main()
