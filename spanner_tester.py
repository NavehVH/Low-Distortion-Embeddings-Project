import random
import csv
import os
from typing import List, Tuple, Dict, Optional
from simple_graph import Graph
from baswana_spanner import visual_spanner
from graph_enums import WeightDistribution, DensityTier


RESULTS_CSV_FILE = "test_results/spanner_test_results.csv"
GRAPHS_DIR = "test_results/graphs"
CONFIGURATIONS_NUMBER = 5
MINIMAL_VERTICES = 100
MAXIMAL_VERTICES = 1000
EXECUTION_NUMBER = 3
STRETCH_FACTORS = [3, 5, 7]
WEIGHT_OPTIONS = [WeightDistribution.UNIFORM, WeightDistribution.EXPONENTIAL, WeightDistribution.NORMAL, WeightDistribution.INTEGER]
DENSITY_TIERS = [DensityTier.SPARSE, DensityTier.MEDIUM, DensityTier.DENSE]

def dijkstra_all_pairs(graph: Graph) -> Dict[int, Dict[int, float]]:
    import heapq
    distances_all = {}
    for start in graph.nodes():
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start] = 0.0
        pq = [(0.0, start)]
        visited = set()
        while pq:
            current_dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    edge_data = graph.adjacency_lists[current][neighbor]
                    edge_weight = edge_data.get('w', 1.0)
                    new_dist = current_dist + edge_weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        distances_all[start] = distances
    return distances_all

def calculate_average_stretch(original_graph: Graph, spanner: Graph, dist_G: Optional[Dict[int,Dict[int,float]]]=None, dist_H: Optional[Dict[int,Dict[int,float]]]=None) -> float:
    original_edges = list(original_graph.edges(data=True))    
    if not original_edges:
        return 1.0
    if dist_G is None:
        dist_G = dijkstra_all_pairs(original_graph)
    if dist_H is None:
        dist_H = dijkstra_all_pairs(spanner)
    total_stretch = 0.0
    valid_edges = 0
    for u, v, _ in original_edges:
        dG = dist_G[u][v]
        dH = dist_H[u][v]
        if dG == float('inf') or dH == float('inf'):
            continue
        stretch = dH / dG if dG > 0 else 1.0
        total_stretch += stretch
        valid_edges += 1
    return total_stretch / valid_edges if valid_edges > 0 else 1.0

def calculate_pair_based_average_stretch(original_graph: Graph, spanner: Graph, dist_G: Optional[Dict[int,Dict[int,float]]]=None, dist_H: Optional[Dict[int,Dict[int,float]]]=None) -> float:
    n = original_graph.number_of_nodes()
    if n <= 1:
        return 1.0
    if dist_G is None:
        dist_G = dijkstra_all_pairs(original_graph)
    if dist_H is None:
        dist_H = dijkstra_all_pairs(spanner)
    nodes = list(original_graph.nodes())
    total = 0.0
    count = 0
    for i in range(len(nodes)):
        u = nodes[i]
        for j in range(i+1, len(nodes)):
            v = nodes[j]
            dG = dist_G[u][v]
            dH = dist_H[u][v]
            if dG == float('inf') or dH == float('inf'):
                continue
            total += (dH / dG) if dG > 0 else 1.0
            count += 1
    return (total / count) if count > 0 else 1.0

def edge_stretch_distribution_counts(original_graph: Graph, spanner: Graph, dist_G: Optional[Dict[int,Dict[int,float]]]=None, dist_H: Optional[Dict[int,Dict[int,float]]]=None) -> Tuple[int,int]:
    original_edges = list(original_graph.edges(data=True))
    if not original_edges:
        return (0, 0)
    if dist_G is None:
        dist_G = dijkstra_all_pairs(original_graph)
    if dist_H is None:
        dist_H = dijkstra_all_pairs(spanner)
    ratios = []
    for u, v, _ in original_edges:
        dG = dist_G[u][v]
        dH = dist_H[u][v]
        if dG == float('inf') or dH == float('inf'):
            continue
        ratios.append(dH / dG if dG > 0 else 1.0)
    if not ratios:
        return (0, 0)
    mean_r = sum(ratios) / len(ratios)
    above = sum(1 for r in ratios if r > mean_r)
    below = sum(1 for r in ratios if r < mean_r)
    return (above, below)

def create_random_graph(n_vertices: int, edge_prob: float, weight_dist: WeightDistribution, weight_params: Tuple = (1, 10), seed: Optional[int] = None) -> Graph:
    if seed is not None:
        random.seed(seed)
    graph = Graph()
    graph.add_nodes_from(range(n_vertices))
    # Create spanning tree for connectivity
    vertices = list(range(n_vertices))
    random.shuffle(vertices)
    for i in range(1, n_vertices):
        graph.add_edge(vertices[random.randint(0, i - 1)], vertices[i], w=weight_dist.generate_weight(weight_params))
    # Add additional random edges
    max_edges = n_vertices * (n_vertices - 1) // 2
    target_edges = int(edge_prob * max_edges)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if not graph.has_edge(i, j) and graph.number_of_edges() < target_edges:
                if random.random() < edge_prob:
                    graph.add_edge(i, j, w= weight_dist.generate_weight(weight_params))
    return graph

def generate_test_configs() -> List[Dict]:
    configs = []
    for i in range(CONFIGURATIONS_NUMBER):
        # Assign density tier (cycle through tiers to ensure variety)
        density_tier = DENSITY_TIERS[i % len(DENSITY_TIERS)]
        # Ensure minimum density for meaningful spanner testing
        edge_prob = max(0.03, density_tier.generate_edge_prob())
        weight_dist = random.choice(WEIGHT_OPTIONS)
        weight_params = weight_dist.generate_params()
        configs.append({
            'n_vertices': random.randint(MINIMAL_VERTICES, MAXIMAL_VERTICES),
            'edge_prob': round(edge_prob, 4),
            'density_tier': density_tier.value,
            'weight_dist': weight_dist,
            'weight_dist_name': weight_dist.name_str,
            'weight_params': weight_params,
            'stretch_factors': STRETCH_FACTORS,
            'graph_seed': random.randint(1, 10000),
            'algo_seed': random.randint(1, 10000)
        })
    return configs

def setup_test_directories() -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV_FILE), exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

# Global counter for test IDs within execution
_test_id_counter = None
def get_next_test_id() -> str:
    global _test_id_counter
    if _test_id_counter is None:
        if not os.path.exists(RESULTS_CSV_FILE):
            _test_id_counter = 1
        else:
            try:
                with open(RESULTS_CSV_FILE, 'r') as f:
                    lines = f.readlines()
                    if len(lines) <= 1:  # Only header or empty
                        _test_id_counter = 1
                    else:
                        # Get the last test ID and start from next number
                        last_line = lines[-1].strip()
                        if last_line:
                            last_id = last_line.split(',')[0]
                            if last_id.startswith('T') and last_id[1:].isdigit():
                                _test_id_counter = int(last_id[1:]) + 1
                            else:
                                _test_id_counter = 1
                        else:
                            _test_id_counter = 1
            except:
                _test_id_counter = 1
    _test_id_counter += 1
    return f"T{_test_id_counter:04d}"

def save_graph_as_text(graph: Graph, filepath: str, metadata: Dict) -> None:
    with open(filepath, 'w') as f:
        f.write(f"# Test ID: {metadata.get('test_id', 'unknown')}\n")
        f.write(f"# Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}\n")
        f.write(f"# Stretch: {metadata.get('stretch', 'N/A')}\n")
        f.write(f"# Graph type: {metadata.get('graph_type', 'unknown')}\n")
        f.write("# Format: node1 node2 weight\n")
        for u, v, attrs in graph.edges(data=True):
            weight = attrs.get('w', 1.0)
            f.write(f"{u} {v} {weight}\n")

def save_to_csv(results: List[Dict]) -> None:
    fields = ['test_id', 'n_vertices', 'original_edges',
              'spanner_edges', 'reduction_rate', 'stretch_factor',
              'average_stretch (Σ(dH(ui,vi)/δG(ui,vi))/|E|)',
              'pair_average_stretch (Σ(δH(u,v)/δG(u,v)) over all pairs / C(n,2))',
              'edge_stretch_above_mean', 'edge_stretch_below_mean',
              'k_parameter',
              'theoretical_bound (k*n^(1+1/k))', 'size_ratio (|E_H|/(k*n^(1+1/k)))',
              'within_bound (spanner_edges<=bound)', 'edge_prob', 'density_percentage', 'density_tier',
              'weight_dist', 'weight_params', 'seed']
    setup_test_directories()
    with open(RESULTS_CSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        file_exists = os.path.exists(RESULTS_CSV_FILE) and os.path.getsize(RESULTS_CSV_FILE) > 0
        if not file_exists:
            writer.writeheader()
        for result in results:
            result_copy = result.copy()
            result_copy.pop('config_idx', None)
            result_copy['theoretical_bound (k*n^(1+1/k))'] = result_copy.pop('theoretical_bound')
            result_copy['within_bound (spanner_edges<=bound)'] = result_copy.pop('within_bound')
            writer.writerow(result_copy)

def run_tests(configs: List[Dict]) -> List[Dict]:
    setup_test_directories()
    results = []
    for config_idx, config in enumerate(configs):
        print(f"  Config {config_idx + 1}/{len(configs)}: n={config['n_vertices']}, {config['density_tier']} density ({config['edge_prob']:.1%}), {config['weight_dist_name']} weights")
        for execution in range(EXECUTION_NUMBER):
            graph = create_random_graph(
                n_vertices=config['n_vertices'],
                edge_prob=config['edge_prob'],
                weight_dist=config['weight_dist'],
                weight_params=config['weight_params'],
                seed=config['graph_seed'] + execution
            )
            for stretch in config['stretch_factors']:
                test_id = get_next_test_id()
                seed = config['algo_seed'] + execution * 100 + stretch
                spanner = visual_spanner(graph, stretch=stretch, seed=seed)
                reduction = (1.0 - spanner.number_of_edges() / max(1, graph.number_of_edges())) * 100
                k = max(1, (stretch + 1) // 2)
                bound = k * (graph.number_of_nodes() ** (1.0 + 1.0/k))
                # precompute distances once per (graph, spanner)
                dist_G = dijkstra_all_pairs(graph)
                dist_H = dijkstra_all_pairs(spanner)
                avg_stretch = calculate_average_stretch(graph, spanner, dist_G, dist_H)
                pair_avg = calculate_pair_based_average_stretch(graph, spanner, dist_G, dist_H)
                edge_above, edge_below = edge_stretch_distribution_counts(graph, spanner, dist_G, dist_H)
                result = {
                    'test_id': test_id,
                    'n_vertices': graph.number_of_nodes(),
                    'original_edges': graph.number_of_edges(),
                    'spanner_edges': spanner.number_of_edges(),
                    'reduction_rate': reduction,
                    'stretch_factor': stretch,
                    'average_stretch (Σ(dH(ui,vi)/δG(ui,vi))/|E|)': round(avg_stretch, 3),
                    'pair_average_stretch (Σ(δH(u,v)/δG(u,v)) over all pairs / C(n,2))': round(pair_avg, 3),
                    'edge_stretch_above_mean': edge_above,
                    'edge_stretch_below_mean': edge_below,
                    'k_parameter': k,
                    'theoretical_bound': int(bound),
                    'size_ratio (|E_H|/(k*n^(1+1/k)))': (spanner.number_of_edges() / bound) if bound > 0 else 0.0,
                    'within_bound': spanner.number_of_edges() <= bound,
                    'seed': seed,
                    'config_idx': config_idx,
                    'edge_prob': config['edge_prob'],
                    'density_percentage': f"{config['edge_prob'] * 100:.1f}%",
                    'density_tier': config['density_tier'],
                    'weight_dist': config['weight_dist_name'],
                    'weight_params': str(config['weight_params'])
                }
                spanner_graph_path = os.path.join(GRAPHS_DIR, f"{test_id}_spanner.txt")
                spanner_metadata = {'test_id': test_id, 'graph_type': 'spanner', 'stretch': stretch}
                save_graph_as_text(spanner, spanner_graph_path, spanner_metadata)
                results.append(result)
    return results

def main():
    print(f"Starting test session: {CONFIGURATIONS_NUMBER} configurations, {EXECUTION_NUMBER} executions each")
    configs = generate_test_configs()
    results = run_tests(configs)
    save_to_csv(results)
    print(f"Completed session - {len(results)} tests saved to {RESULTS_CSV_FILE}")

if __name__ == "__main__":
    main()
