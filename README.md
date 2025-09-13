# Baswana-Sen Spanner Algorithm Implementation

A Python implementation of the **Baswana-Sen algorithm** for computing **(2k-1)-spanners** in weighted graphs, based on the seminal 2003 paper "A Simple and Linear Time Randomized Algorithm for Computing Sparse Spanners in Weighted Graphs" by Surender Baswana and Sandeep Sen.

## Academic Background

This implementation is based on the groundbreaking research by Baswana and Sen that introduced the first linear-time randomized algorithm for computing sparse spanners. Their algorithm achieves:

- **Time Complexity**: O(km) expected time
- **Spanner Size**: O(kn^(1+1/k)) edges  
- **Stretch Factor**: (2k-1)
- **Key Innovation**: A novel clustering approach that avoids distance computations entirely

The algorithm's size bound essentially matches the worst-case lower bound implied by the 43-year-old girth conjecture made independently by Erdős, Bollobás, and Bondy & Simonovits.


## Repository Structure

```
baswana_spanner.py      # Main Baswana-Sen algorithm implementation
simple_graph.py         # Custom Graph class with adjacency list representation
README.md              # This documentation
```

## Requirements

- **Python 3.8+**
- No external dependencies required

## Algorithm Overview

The Baswana-Sen algorithm operates in two phases:

### Phase 1: Cluster Formation (k-1 iterations)
1. **Sampling**: Each cluster is sampled with probability n^(-1/k)
2. **Vertex Assignment**: Non-sampled vertices join their nearest sampled neighbor
3. **Edge Selection**: Carefully chosen edges are added to the spanner
4. **Clustering Update**: New clustering is formed for the next iteration

### Phase 2: Vertex-Cluster Joining
- Each vertex connects to its neighboring clusters via lightest edges
- Ensures the final stretch bound of (2k-1)

## Usage

Run the interactive program:

```bash
python3 baswana_spanner.py
```

### Interactive Input Process

1. **Number of edges**: Enter total edges in your graph
2. **Edge specification**: For each edge, enter: `vertex1 vertex2 weight`
3. **Stretch factor**: Enter desired stretch (algorithm computes k = ⌊(stretch+1)/2⌋)
4. **Random seed**: Enter seed for reproducible results

### Complete Example Session with Repeat Options

The program provides two levels of repetition to explore different aspects of the algorithm:

#### 1. Same Graph, Different Random Seeds
After computing a spanner, you can run the algorithm again on the **same graph** with **different randomization** to see how the random sampling affects the result:

```
Enter number of edges: 4
Enter each edge as: vertex1 vertex2 weight (Example: 0 1 1.0)
Edge 1: 0 1 2.5
Edge 2: 1 2 1.0
Edge 3: 2 3 3.0
Edge 4: 0 3 4.0
Enter stretch factor: 3
Enter random seed: 42

Original edges: [(0, 1, 2.5), (1, 2, 1.0), (2, 3, 3.0), (0, 3, 4.0)]
Spanner edges: [(0, 1, 2.5), (1, 2, 1.0), (2, 3, 3.0)]
k = 2 | Edge reduction: 25.00%

Run again with different seed? (y/n): y

[Algorithm runs again with seed 43]
Original edges: [(0, 1, 2.5), (1, 2, 1.0), (2, 3, 3.0), (0, 3, 4.0)]
Spanner edges: [(0, 1, 2.5), (2, 3, 3.0), (0, 3, 4.0)]
k = 2 | Edge reduction: 25.00%

Run again with different seed? (y/n): n
```

#### 2. Completely New Graph and Parameters
After finishing with one graph configuration, you can start over with a **completely different graph**:

```
Run again with different parameters? (y/n): y

[Program restarts - you can now enter a completely new graph]
Enter number of edges: 3
Enter each edge as: vertex1 vertex2 weight (Example: 0 1 1.0)
Edge 1: 0 1 1.0
Edge 2: 1 2 2.0  
Edge 3: 0 2 5.0
Enter stretch factor: 5
Enter random seed: 100

[Results for the new graph...]
```

### Practical Usage Tips

1. **For algorithm demonstration**: Use the same graph with different seeds (option 1) to show randomization effects
2. **For comprehensive testing**: Use different graphs (option 2) to explore various scenarios
3. **For reproducible results**: Use the same seed across runs to get identical results
4. **For comparison studies**: Run multiple seeds on the same graph to compare different valid spanners

## Visualization Suite

The repository also includes a visualizer (`viz_spanner.py`) that produces **side-by-side plots** of the original graph and its computed spanner.

All images are saved under the `outputs/` directory, organized per test case.

### Run the Entire Suite

Run all built-in test cases which are listed in experiments.txt (multiple graphs and stretch factors)

```bash
MPLBACKEND=Agg python3 viz_spanner.py suite --name all
```

## Advanced Testing Framework

### Spanner Tester (`spanner_tester.py`)

A comprehensive testing framework for evaluating the Baswana-Sen algorithm on randomly generated graphs with various parameters:

#### Features:
- **Random graph generation** with configurable vertex count, edge probability, and weight distributions
- **Multiple test configurations** with different density tiers and weight patterns
- **Automated spanner analysis** including theoretical bound verification
- **Average stretch calculation** using the formula: `Σ(dH(ui,vi)/w(ui,vi)) / |E|` where H is the spanner graph
- **CSV output** with comprehensive metrics and test metadata
- **Graph serialization** for result reproducibility

#### Configuration Constants:
- `MINIMAL_VERTICES = 100`: Minimum graph size for testing
- `MAXIMAL_VERTICES = 1000`: Maximum graph size for testing
- `CONFIGURATIONS_NUMBER = 5`: Number of different test configurations per run
- `EXECUTION_NUMBER = 3`: Repetitions per configuration for statistical reliability
- `STRETCH_FACTORS = [3, 5, 7]`: Stretch factors to test

#### Usage:
```bash
python3 spanner_tester.py
```

Results are saved in `test_results/spanner_test_results.csv` with detailed metrics for analysis.

### Graph Configuration Enums (`graph_enums.py`)

Defines standardized graph generation parameters:

#### Weight Distributions:
- **UNIFORM**: Uniform random weights within specified range
- **EXPONENTIAL**: Exponentially distributed weights (natural for network distances)
- **NORMAL**: Normally distributed weights (good for physical networks)
- **INTEGER**: Integer weights only (discrete scenarios)

#### Density Tiers:
- **SPARSE**: Low edge probability (3-15%) - models sparse networks
- **MEDIUM**: Medium edge probability (20-40%) - balanced connectivity
- **DENSE**: High edge probability (40-60%) - highly connected graphs

Each enum provides methods for parameter generation and weight sampling, ensuring consistent and reproducible graph generation across test runs.
