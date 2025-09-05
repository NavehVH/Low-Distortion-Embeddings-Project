# Low-Distortion-Embeddings-Project Baswana–Sen Spanner (Python)

Small, dependency-light Python project that builds a **(2k–1)-spanner** from an input graph using the Baswana–Sen algorithm.

## Repo layout

    simple_graph.py                    # minimal undirected graph + helpers
    spanner.py                         # Baswana–Sen implementation with CLI
    baswana_sen_spanner.py             # Baswana–Sen implementation

## Requirements

- Python 3.8+

No external dependencies required.

## Quick start

Run the algorithm on an example graph:

    python3 spanner.py --graph example --stretch 3

This will build a spanner and print statistics including:
- Number of nodes and edges in original vs spanner graph
- Edge reduction percentage
- Stretch guarantee verification

## CLI usage

    python3 spanner.py
      --graph {example,random,edgelist}
      [--edgelist PATH]             # path to edgelist file for --graph edgelist
      [--n N --m M]                 # only for --graph random
      [--weighted]                  # only for --graph random
      [--stretch T]                 # desired stretch; algorithm uses k=floor((t+1)/2)
      [--seed SEED]                 # integer for reproducible randomness
      [--print-edges]               # print all spanner edges

## Examples

Run algorithm on built-in example graph:

    python3 spanner.py --graph example --stretch 3 --print-edges

Generate and process a random graph:

    python3 spanner.py --graph random --n 50 --m 100 --weighted --stretch 5 --seed 42

Process graph from edgelist file:

    python3 spanner.py --graph edgelist --edgelist my_graph.txt --stretch 3

## Input formats

**Edgelist format** (for `--graph edgelist`):
- Unweighted: `u v` (one edge per line)
- Weighted: `u v w` (one edge per line with weight)

## Notes

- **Stretch vs k:** the algorithm outputs a \((2k-1)\)-spanner with \(k = \lfloor (t+1)/2 \rfloor\).  
  Example: `--stretch 3` → `k=2` → 3-spanner.
- **Randomization:** Baswana–Sen is randomized. Different seeds can produce different valid spanners. Using `--seed` makes the result reproducible; omitting it uses fresh randomness.
- **Algorithm verification:** The implementation includes automatic stretch guarantee verification for all node pairs.
