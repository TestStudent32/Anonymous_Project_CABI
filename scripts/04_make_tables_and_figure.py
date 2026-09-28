"""Step 4: every number, table and the figure in the paper, from the CSVs in results/.

Prints: Table 1 (system level, H=1), Table 2 (R^2 by horizon), the naive-baseline horizon check,
validation-selected model counts, and the station-level TimesFM comparison with paired Wilcoxon
signed-rank tests. Saves figures/horizon.png (Fig. 1) with matplotlib.

Usage:  python scripts/04_make_tables_and_figure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import config  # noqa: E402
from src.selection import validation_selected  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
R = config.RESULTS_DIR
H = config.HORIZONS
REF = "Polynomial Regression"  # system-level reference model (highest validation R^2 at every horizon; checked below)


def system(h, phase="test"):
    d = pd.read_csv(R / f"system_feature_models_h{h}.csv")
    return d[d["phase"] == phase]


def station(h):
    d = pd.read_csv(R / f"station_feature_models_h{h}.csv", dtype={"station_id": str})
    return validation_selected(d)


def main():
    tfm_sys = pd.read_csv(R / "timesfm_system.csv")
    tfm_st = pd.read_csv(R / "timesfm_station.csv", dtype={"station_id": str})

    # ---- Table 1: system level, H = 1 ----
    t1 = system(1).pivot_table(index="model", columns="target", values=["R2", "MAE"])
    tf1 = tfm_sys[tfm_sys["horizon"] == 1].set_index("target")
    t1.loc["TimesFM (zero-shot)"] = [tf1.loc[c[1], c[0]] for c in t1.columns]
    print("Table 1 - system level, H=1 (test 2025)\n", t1.sort_values(("R2", "incoming_trips"), ascending=False).round(3), "\n")

    # ---- Table 2 + figure data ----
    rows = {"System, polynomial (in)": [], "System, TimesFM (in)": [],
            "Station, validation-selected (mean)": [], "Station, TimesFM (mean)": []}
    print("Station level: validation-selected feature model vs TimesFM (94 series)")
    for h in H:
        s = system(h)
        rows["System, polynomial (in)"].append(s[(s.model == REF) & (s.target == "incoming_trips")]["R2"].iloc[0])
        t = tfm_sys[(tfm_sys.horizon == h) & (tfm_sys.target == "incoming_trips")]
        rows["System, TimesFM (in)"].append(t["R2"].iloc[0])
        v = system(h, "val")
        best_val = v.loc[v.groupby("target")["R2"].idxmax()].set_index("target")["model"].to_dict()
        print(f"  [check] H={h}: best system model on validation = {best_val}")

        sel = station(h)
        m = sel.merge(tfm_st[tfm_st.horizon == h], on=["station_id", "target"], suffixes=("_feat", "_tfm"))
        rows["Station, validation-selected (mean)"].append(m["R2_feat"].mean())
        rows["Station, TimesFM (mean)"].append(m["R2_tfm"].mean())
        print(f"  H={h:>2}: n={len(m)}  mean R2 feature {m['R2_feat'].mean():.3f} vs TimesFM {m['R2_tfm'].mean():.3f}; "
              f"TimesFM higher in {(m['R2_tfm'] > m['R2_feat']).sum()}/{len(m)}; "
              f"Wilcoxon p (R2) = {wilcoxon(m['R2_tfm'], m['R2_feat']).pvalue:.2g}, "
              f"p (MAE) = {wilcoxon(m['MAE_tfm'], m['MAE_feat']).pvalue:.2g}; "
              f"selected models: {sel['model'].value_counts().head(3).to_dict()}")
    print("\nTable 2 - test R^2 by horizon\n", pd.DataFrame(rows, index=H).T.round(3), "\n")

    # ---- Correctness check of the horizon rule: naive baseline must not change with H ----
    naive = [system(h).query("model == 'Naive (same hour last week)'")["R2"].round(4).tolist() for h in H]
    print("Naive-baseline R^2 by horizon (must be identical):", naive)

    legacy = R / "timesfm_system_legacy.csv"
    if legacy.exists():
        print("\nLegacy TimesFM protocol:\n", pd.read_csv(legacy)[["protocol", "target", "R2"]].round(3))

    # ---- Figure 1 ----
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=300)
    styles = [("System, polynomial (in)", blue, "-", "o", "System-wide, polynomial"),
              ("System, TimesFM (in)", blue, "--", "s", "System-wide, TimesFM (zero-shot)"),
              ("Station, validation-selected (mean)", orange, "-", "o", "Station mean, validation-selected model"),
              ("Station, TimesFM (mean)", orange, "--", "s", "Station mean, TimesFM (zero-shot)")]
    for key, color, ls, marker, label in styles:
        ax.plot(H, rows[key], color=color, linestyle=ls, marker=marker, markersize=4, linewidth=1.8, label=label)
    ax.set(xlabel="Forecast horizon $H$ (hours ahead)", ylabel="$R^2$ (test, 2025)", xticks=H, ylim=(0.4, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.legend(frameon=False, loc="center right", fontsize=7)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "horizon.png")
    print(f"\n-> {config.FIGURES_DIR / 'horizon.png'}")


if __name__ == "__main__":
    main()
