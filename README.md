# Low-Distortion-Embeddings-Project Baswana–Sen Spanner (Python)

Small, dependency-light Python project that:
- Builds a **(2k–1)-spanner** from an input graph using the Baswana–Sen algorithm.
- Visualizes **Original graph G** vs **Spanner H** side-by-side with `|V|`, `|E|`, and edge-reduction %.

## Repo layout

    simple_graph.py   # minimal undirected graph + helpers
    spanner.py        # Baswana–Sen implementation: baswana_sen_spanner(...)
    viz_spanner.py    # CLI to build a demo graph, run spanner, and save PNG
    baswana_sen_spanne.py # Just the code with comments

## Requirements

- Python 3.8+
- matplotlib

Install:

    pip3 install matplotlib

## Quick start (triangle demo)

Headless save to PNG (works on servers/WSL/containers):

    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case triangle --stretch 3 --seed 1 --save ex_triangle.png

This writes `ex_triangle.png` with two panels:
- Left: original **G**
- Right: spanner **H** (shows `|V|`, `|E|`, and `ΔE` edge-reduction %)

> On a desktop with a GUI backend installed, you can omit `MPLBACKEND=Agg` and `--save` to open an interactive window.

## CLI usage

    python3 viz_spanner.py
      --graph {example,random}
      [--case {triangle,square,square_diag,square_diag_equal,path6,star6,k4_mix,k4_complete,two_tris_bridge,ladder,pentagon,wheel6}]
      [--n N --m M]                 # only for --graph random
      [--weighted]                  # only for --graph random
      [--stretch T]                 # desired t; algorithm uses k=floor((t+1)/2)
      [--seed SEED]                 # integer for reproducible randomness; omit for fresh randomness (Use 1)
      [--save PATH.png]             # save figure; otherwise shows a window (if GUI backend exists)

### Example graphs (when `--graph example`)

- `triangle` – 3 nodes, one heavy edge (two edges weight 1.0, one edge weight 2.0)
- `square` – 4-cycle (unweighted)
- `square_diag` – 4-cycle + diagonal (diagonal heavier)
- `square_diag_equal` – 4-cycle + diagonal, **all weight = 1.0**
- `path6` – path on 6 nodes (tree)
- `star6` – star with 5 leaves (tree)
- `k4_mix` – K4 with mixed weights (triangle light, edges to 4th node heavier)
- `k4_complete` – K4 unweighted
- `two_tris_bridge` – two triangles joined by a light bridge
- `ladder` – 2×4 ladder (rails light, rungs heavier)
- `pentagon` – 5-cycle (unweighted)
- `wheel6` – wheel (center + 5 rim nodes; spokes light, rim heavier)

Run any case (seeded, headless save):

    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case wheel6 --stretch 3 --seed 1 --save ex_wheel6.png

## Run ALL built-in examples (seed = 1)

One-by-one:

    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case triangle            --stretch 3 --seed 1 --save ex_triangle.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case square              --stretch 3 --seed 1 --save ex_square.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case square_diag         --stretch 3 --seed 1 --save ex_square_diag.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case square_diag_equal   --stretch 3 --seed 1 --save ex_square_diag_equal.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case path6               --stretch 3 --seed 1 --save ex_path6.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case star6               --stretch 3 --seed 1 --save ex_star6.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case k4_mix              --stretch 3 --seed 1 --save ex_k4_mix.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case k4_complete         --stretch 3 --seed 1 --save ex_k4_complete.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case two_tris_bridge     --stretch 3 --seed 1 --save ex_bridge.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case ladder              --stretch 3 --seed 1 --save ex_ladder.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case pentagon            --stretch 3 --seed 1 --save ex_pentagon.png
    MPLBACKEND=Agg python3 viz_spanner.py --graph example --case wheel6              --stretch 3 --seed 1 --save ex_wheel6.png

Bash loop:

    cases=(triangle square square_diag square_diag_equal path6 star6 k4_mix k4_complete two_tris_bridge ladder pentagon wheel6)
    for c in "${cases[@]}"; do
      MPLBACKEND=Agg python3 viz_spanner.py --graph example --case "$c" --stretch 3 --seed 1 --save "ex_${c}.png"
    done

## Random graphs

    MPLBACKEND=Agg python3 viz_spanner.py --graph random --n 80 --m 200 --weighted --stretch 5 --seed 42 --save ex_random.png

## Notes

- **Stretch vs k:** the algorithm outputs a \((2k-1)\)-spanner with \(k = \lfloor (t+1)/2 \rfloor\).  
  Example: `--stretch 3` → `k=2` → 3-spanner.
- **Randomization:** Baswana–Sen is randomized. Different seeds can produce different valid spanners. Using `--seed` makes the result reproducible; omitting it uses fresh randomness.
- **Headless environments:** If you see Qt/tk warnings, keep `MPLBACKEND=Agg` and `--save` to write a PNG instead of opening a window.

## Optional: print spanner edges in console

If `spanner.py` exposes a small CLI, for example:

    python3 spanner.py --graph example --stretch 3 --print-edges

This prints the selected spanner edges and verifies the stretch guarantee.
