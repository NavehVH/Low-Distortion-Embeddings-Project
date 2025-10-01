
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "test_results/spanner_test_results.csv"
OUT_DIR = "test_results/graphs"

# Column names as saved by spanner_tester.py
COL_TEST_ID = "test_id"
COL_N = "n_vertices"
COL_M = "original_edges"
COL_MH = "spanner_edges"
COL_REDUCT = "reduction_rate"  # percentage already
COL_T = "stretch_factor"
COL_AVG_EDGE = "average_stretch (Σ(dH(ui,vi)/δG(ui,vi))/|E|)"
COL_AVG_PAIR = "pair_average_stretch (Σ(δH(u,v)/δG(u,v)) over all pairs / C(n,2))"
COL_ABOVE = "edge_stretch_above_mean"
COL_BELOW = "edge_stretch_below_mean"
COL_K = "k_parameter"
COL_BOUND = "theoretical_bound (k*n^(1+1/k))"
COL_SIZE_RATIO = "size_ratio (|E_H|/(k*n^(1+1/k)))"
COL_WITHIN = "within_bound (spanner_edges<=bound)"
COL_EDGE_PROB = "edge_prob"
COL_DENSITY_PCT = "density_percentage"
COL_DENSITY_TIER = "density_tier"
COL_WEIGHT_DIST = "weight_dist"
COL_WEIGHT_PARAMS = "weight_params"
COL_SEED = "seed"

def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def _load_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[COL_T]).copy()
    for c in [COL_DENSITY_TIER, COL_WEIGHT_DIST]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _agg_mean_std(df, metric, by=[COL_T]):
    g = df.groupby(by, dropna=True)[metric].agg(['mean', 'std']).reset_index().sort_values(by)
    return g

def _plot_mean_std(df_agg, xcol, ymean, ystd, title, xlab, ylab, outpath):
    fig = plt.figure()
    x = df_agg[xcol].values
    y = df_agg[ymean].values
    yerr = df_agg[ystd].fillna(0.0).values
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linestyle='-')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

def _plot_lines_by_category(df, metric, category_col, title, ylab, filename_base):
    fig = plt.figure()
    for cat, sub in sorted(df.groupby(category_col)):
        agg = _agg_mean_std(sub, metric, by=[COL_T])
        if agg.empty:
            continue
        x = agg[COL_T].values
        y = agg['mean'].values
        yerr = agg['std'].fillna(0.0).values
        plt.plot(x, y, marker='o', label=str(cat))
        try:
            import numpy as np
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)
        except Exception:
            pass
    plt.title(title)
    plt.xlabel("Stretch factor (t)")
    plt.ylabel(ylab)
    plt.grid(True, linestyle="--", alpha=0.5)
    if df[category_col].nunique() > 1:
        plt.legend(title=category_col, loc="best")
    outpath = os.path.join(OUT_DIR, f"{filename_base}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath

def generate_all():
    _ensure_out_dir()
    df = _load_csv()
    outputs = []

    # 1) Average edge-based stretch vs t (+ variants)
    if "average_stretch (Σ(dH(ui,vi)/δG(ui,vi))/|E|)" in df.columns:
        col = "average_stretch (Σ(dH(ui,vi)/δG(ui,vi))/|E|)"
        agg = _agg_mean_std(df, col, by=[COL_T])
        if not agg.empty:
            out = os.path.join(OUT_DIR, "avg_edge_stretch_vs_t.png")
            _plot_mean_std(agg, COL_T, 'mean', 'std',
                           "Average Edge-Based Stretch vs Stretch Factor (t)",
                           "Stretch factor (t)",
                           "Average edge stretch",
                           out)
            outputs.append(out)
        if COL_DENSITY_TIER in df.columns:
            outputs.append(_plot_lines_by_category(df, col, COL_DENSITY_TIER,
                                                   "Avg Edge-Based Stretch vs t (by density tier)",
                                                   "Average edge stretch",
                                                   "avg_edge_stretch_vs_t_by_density"))
        if COL_WEIGHT_DIST in df.columns:
            outputs.append(_plot_lines_by_category(df, col, COL_WEIGHT_DIST,
                                                   "Avg Edge-Based Stretch vs t (by weight dist)",
                                                   "Average edge stretch",
                                                   "avg_edge_stretch_vs_t_by_weight"))

    # 1b) Pair-based average stretch vs t
    if "pair_average_stretch (Σ(δH(u,v)/δG(u,v)) over all pairs / C(n,2))" in df.columns:
        col = "pair_average_stretch (Σ(δH(u,v)/δG(u,v)) over all pairs / C(n,2))"
        agg = _agg_mean_std(df, col, by=[COL_T])
        if not agg.empty:
            out = os.path.join(OUT_DIR, "avg_pair_stretch_vs_t.png")
            _plot_mean_std(agg, COL_T, 'mean', 'std',
                           "Average Pair-Based Stretch vs Stretch Factor (t)",
                           "Stretch factor (t)",
                           "Average pair stretch",
                           out)
            outputs.append(out)

    # 2) Edge reduction vs t
    if "reduction_rate" in df.columns:
        col = "reduction_rate"
        agg = _agg_mean_std(df, col, by=[COL_T])
        if not agg.empty:
            out = os.path.join(OUT_DIR, "edge_reduction_vs_t.png")
            _plot_mean_std(agg, COL_T, 'mean', 'std',
                           "Edge Reduction vs Stretch Factor (t)",
                           "Stretch factor (t)",
                           "Edge reduction (%)",
                           out)
            outputs.append(out)
        if COL_DENSITY_TIER in df.columns:
            outputs.append(_plot_lines_by_category(df, col, COL_DENSITY_TIER,
                                                   "Edge Reduction vs t (by density tier)",
                                                   "Edge reduction (%)",
                                                   "edge_reduction_vs_t_by_density"))
        if COL_WEIGHT_DIST in df.columns:
            outputs.append(_plot_lines_by_category(df, col, COL_WEIGHT_DIST,
                                                   "Edge Reduction vs t (by weight dist)",
                                                   "Edge reduction (%)",
                                                   "edge_reduction_vs_t_by_weight"))

    # 3) Size ratio vs theoretical bound vs t
    metric_col = None
    if "size_ratio (|E_H|/(k*n^(1+1/k)))" in df.columns:
        metric_col = "size_ratio (|E_H|/(k*n^(1+1/k)))"
    elif ("spanner_edges" in df.columns) and ("theoretical_bound (k*n^(1+1/k))" in df.columns):
        df["__size_ratio__"] = df["spanner_edges"] / df["theoretical_bound (k*n^(1+1/k))"].replace({0: float('nan')})
        metric_col = "__size_ratio__"

    if metric_col is not None:
        agg = _agg_mean_std(df, metric_col, by=[COL_T])
        if not agg.empty:
            out = os.path.join(OUT_DIR, "size_ratio_vs_bound_vs_t.png")
            _plot_mean_std(agg, COL_T, 'mean', 'std',
                           "Size Ratio vs Theoretical Bound (n^{1+1/k})",
                           "Stretch factor (t)",
                           "Size ratio (|E_S| / (k·n^{1+1/k}))",
                           out)
            outputs.append(out)
        if COL_DENSITY_TIER in df.columns:
            outputs.append(_plot_lines_by_category(df, metric_col, COL_DENSITY_TIER,
                                                   "Size Ratio vs Bound (by density tier)",
                                                   "Size ratio (|E_S| / (k·n^{1+1/k}))",
                                                   "size_ratio_vs_bound_vs_t_by_density"))
        if COL_WEIGHT_DIST in df.columns:
            outputs.append(_plot_lines_by_category(df, metric_col, COL_WEIGHT_DIST,
                                                   "Size Ratio vs Bound (by weight dist)",
                                                   "Size ratio (|E_S| / (k·n^{1+1/k}))",
                                                   "size_ratio_vs_bound_vs_t_by_weight"))

    report = os.path.join(OUT_DIR, "graphs_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        for p in outputs:
            f.write(str(p) + "\n")
    print("Generated files:")
    for p in outputs:
        print(" -", p)

if __name__ == "__main__":
    generate_all()
