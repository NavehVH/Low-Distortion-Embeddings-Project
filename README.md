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
