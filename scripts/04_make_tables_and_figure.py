"""Step 4: every table, statistic and figure in the paper, generated from the CSVs in results/.

Writes LaTeX table bodies to tables/*.tex (the paper \\input{}s them, so no number is retyped), the
figures to figures/*.png (matplotlib), and prints the statistics quoted in the text.

  tables/system_h1.tex        Table: system level, H=1, all models (test 2025)
  tables/horizons.tex         Table: system-level R^2 by horizon, both test years
  tables/stations.tex         Table: station level, mean R^2 and paired Wilcoxon tests vs the feature model
  tables/forecast_weather.tex Table: observed vs day-ahead forecast weather at the target hour
  tables/ablation.tex         Table: system-level feature ablation by horizon
  figures/horizon.png         Fig: R^2 vs horizon, system and station level
  figures/station_gain.png    Fig: per-series R^2 difference (Chronos-2 + covariates minus feature model), H=8

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
TAB = config.ROOT / "tables"
FM = [("timesfm", "TimesFM"), ("chronos_bolt", "Chronos-Bolt"),
      ("chronos2_nocov", "Chronos-2"), ("chronos2", "Chronos-2 + cov.")]
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#4a3aa7"


# ---------------------------------------------------------------- loading helpers
def system(h, root=R):
    return pd.read_csv(root / f"system_feature_models_h{h}.csv")


def system_selected(h, target, root=R):
    """Test row of the system model with the best validation R^2 (same selection rule as per station)."""
    d = system(h, root)
    d = d[d.target == target]
    v = d[d.phase == "val"]
    name = v.loc[v.R2.idxmax(), "model"]
    return d[(d.phase == "test") & (d.model == name)].iloc[0]


def fm_system(slug, root=R):
    return pd.read_csv(root / f"{slug}_system.csv")


def station_selected(h, root=R):
    return validation_selected(pd.read_csv(root / f"station_feature_models_h{h}.csv", dtype={"station_id": str}))


def fm_station(slug, root=R):
    return pd.read_csv(root / f"{slug}_station.csv", dtype={"station_id": str})


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def write(name, lines):
    TAB.mkdir(exist_ok=True)
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> tables/{name}")


# ---------------------------------------------------------------- tables
def table_system_h1():
    d = system(1)
    t = d[d.phase == "test"].pivot_table(index="model", columns="target", values=["R2", "MAE"])
    rows = [(m, t.loc[m, ("R2", "incoming_trips")], t.loc[m, ("R2", "outgoing_trips")],
             t.loc[m, ("MAE", "incoming_trips")], t.loc[m, ("MAE", "outgoing_trips")]) for m in t.index]
    for slug, name in FM:
        f = fm_system(slug).set_index(["target", "horizon"])
        rows.append((name + " (zero-shot)", f.loc[("incoming_trips", 1), "R2"], f.loc[("outgoing_trips", 1), "R2"],
                     f.loc[("incoming_trips", 1), "MAE"], f.loc[("outgoing_trips", 1), "MAE"]))
    rows.sort(key=lambda r: -r[1])
    lines = [f"{n} & {f3(a)} / {f3(b)} & {c:.1f} / {e:.1f} \\\\" for n, a, b, c, e in rows]
    write("system_h1.tex", lines)
    print(pd.DataFrame(rows, columns=["model", "R2 in", "R2 out", "MAE in", "MAE out"]).round(3).to_string(index=False))


def table_horizons():
    lines = []
    for year, root in (("2025", R), ("2024", R / "test2024")):
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{Test year {year}}}}} \\\\")
        sel = [system_selected(h, "incoming_trips", root) for h in H]
        lines.append("Feature model (val.-selected) & " + " & ".join(f3(s.R2) for s in sel) + " \\\\")
        for slug, name in FM:
            f = fm_system(slug, root)
            f = f[f.target == "incoming_trips"].set_index("horizon").R2
            lines.append(f"{name} & " + " & ".join(f3(f[h]) for h in H) + " \\\\")
        print(f"  {year}: selected feature models = {[s.model for s in sel]}")
    write("horizons.tex", lines)


def table_stations():
    """Mean R^2 per horizon; markers from paired two-sided Wilcoxon tests vs the feature model."""
    lines = []
    feat = {h: station_selected(h) for h in H}
    lines.append("Feature model (val.-selected) & " + " & ".join(f3(feat[h].R2.mean()) for h in H) + " \\\\")
    for slug, name in FM:
        cells = []
        for h in H:
            m = feat[h].merge(fm_station(slug)[lambda d: d.horizon == h], on=["station_id", "target"], suffixes=("_f", "_m"))
            p = wilcoxon(m.R2_m, m.R2_f).pvalue
            mark = "" if p >= 0.01 else ("$^{+}$" if m.R2_m.mean() > m.R2_f.mean() else "$^{-}$")
            cells.append(f3(m.R2_m.mean()) + mark)
            print(f"  station {name:<17} H={h:>2}: mean {m.R2_m.mean():.3f} vs {m.R2_f.mean():.3f}, "
                  f"higher in {(m.R2_m > m.R2_f).sum()}/{len(m)}, p={p:.1g}; "
                  f"MAE {m.MAE_m.mean():.3f} vs {m.MAE_f.mean():.3f}, lower in {(m.MAE_m < m.MAE_f).sum()}/{len(m)}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    write("stations.tex", lines)
    for h in (1, 8):
        print(f"  validation-selected station models, H={h}: {feat[h].model.value_counts().head(4).to_dict()}")


def table_forecast_weather():
    lines = []
    for h in (1, 8, 24):
        s = system(h, R / "forecast_weather") if (R / "forecast_weather" / f"system_feature_models_h{h}.csv").exists() else None
        obs = system_selected(h, "incoming_trips")
        fw = s[(s.model == obs.model) & (s.target == "incoming_trips") & (s.phase == "test_forecast_weather")].R2.iloc[0]
        c2 = fm_system("chronos2").query("target == 'incoming_trips' and horizon == @h").R2.iloc[0]
        c2f = fm_system("chronos2_fcstweather").query("target == 'incoming_trips' and horizon == @h").R2.iloc[0]
        lines.append(f"System, $H{{=}}{h}$ & {f3(obs.R2)} & {f3(fw)} & {f3(c2)} & {f3(c2f)} \\\\")
    st = R / "forecast_weather" / "station_feature_models_h8.csv"
    if st.exists():
        d = pd.read_csv(st, dtype={"station_id": str})
        sel = validation_selected(d[d.phase.isin(["val", "test"])])
        fw = d[d.phase == "test_forecast_weather"].merge(sel[["station_id", "target", "model"]])
        c2 = fm_station("chronos2").query("horizon == 8").R2.mean()
        c2f_path = R / "chronos2_fcstweather_station.csv"
        c2f = "--"
        if c2f_path.exists():
            f = fm_station("chronos2_fcstweather").query("horizon == 8")
            if f.station_id.nunique() == sel.station_id.nunique():  # never report a partial run
                c2f = f3(f.R2.mean())
            else:
                print(f"  (Chronos-2 forecast-weather station run incomplete: {f.station_id.nunique()} stations)")
        lines.append(f"Station mean, $H{{=}}8$ & {f3(sel.R2.mean())} & {f3(fw.R2.mean())} & {f3(c2)} & {c2f} \\\\")
    write("forecast_weather.tex", lines)


def table_ablation():
    d = pd.read_csv(R / "ablation_system.csv").query("target == 'incoming_trips'")
    order = ["calendar", "calendar+weather", "demand", "demand+calendar", "all"]
    p = d.pivot(index="subset", columns="horizon", values="test_R2").loc[order]
    lines = [f"{s.replace('+', ' + ')} & " + " & ".join(f3(p.loc[s, h]) for h in H) + " \\\\" for s in order]
    write("ablation.tex", lines)


def table_volume(h=8):
    """Does the per-series gain of Chronos-2 + covariates depend on station volume? Volume = mean trips per
    hour and direction in the test year; series grouped into volume terciles."""
    from scipy.stats import spearmanr
    m = station_selected(h).merge(fm_station("chronos2").query("horizon == @h"),
                                  on=["station_id", "target"], suffixes=("_f", "_m"))
    m["gain"] = m.R2_m - m.R2_f
    vol = {}
    for sid in m.station_id.unique():
        s = pd.read_csv(config.STATION_DIR / f"station_{sid}.csv", usecols=["datetime"] + config.TARGETS,
                        parse_dates=["datetime"])
        s = s[s.datetime.dt.year == config.TEST_YEAR]
        vol[sid] = s[config.TARGETS].to_numpy().sum() / 2 / 8760
    m["vol"] = m.station_id.map(vol)
    rho = spearmanr(m.vol, m.gain)
    m["tier"] = pd.qcut(m.vol, 3, labels=["Low", "Middle", "High"])
    g = m.groupby("tier", observed=True).agg(n=("gain", "size"), vol=("vol", "mean"), feat=("R2_f", "mean"),
                                             fm=("R2_m", "mean"), gain=("gain", "mean"), wins=("gain", lambda x: (x > 0).sum()))
    lines = [f"{t} & {r.vol:.1f} & {f3(r.feat)} & {f3(r.fm)} & {f'{r.gain:+.3f}'.replace('-', '$-$')} & "
             f"{int(r.wins)}/{int(r.n)} \\\\" for t, r in g.iterrows()]
    write("volume.tex", lines)
    print(f"  Spearman(volume, gain) = {rho.correlation:.2f}, p = {rho.pvalue:.1g}")
    print(g.round(3).to_string())


# ---------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)


def figure_horizon():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)
    series = [("Feature model (val.-selected)", "-", "o", GREY)] + \
             [(n, "-" if s == "chronos2" else "--", "s", c) for (s, n), c in zip(FM, [ORANGE, GREEN, PURPLE, BLUE])]
    sys_vals = {"Feature model (val.-selected)": [system_selected(h, "incoming_trips").R2 for h in H]}
    st_vals = {"Feature model (val.-selected)": [station_selected(h).R2.mean() for h in H]}
    for slug, name in FM:
        f = fm_system(slug).query("target == 'incoming_trips'").set_index("horizon").R2
        sys_vals[name] = [f[h] for h in H]
        g = fm_station(slug).groupby("horizon").R2.mean()
        st_vals[name] = [g[h] for h in H]
    for ax, vals, title in ((axes[0], sys_vals, "System-wide (incoming)"), (axes[1], st_vals, "Station mean (94 series)")):
        for (name, ls, mk, c) in series:
            ax.plot(H, vals[name], linestyle=ls, marker=mk, color=c, markersize=3.5, linewidth=1.6, label=name)
        ax.set(title=title, xlabel="Forecast horizon $H$ (h)", xticks=H)
        style(ax)
    axes[0].set_ylabel("$R^2$ (test 2025)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # leave room for the shared legend below the panels
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "horizon.png")
    print("  -> figures/horizon.png")


def figure_station_gain():
    feat = station_selected(8)
    m = feat.merge(fm_station("chronos2").query("horizon == 8"), on=["station_id", "target"], suffixes=("_f", "_m"))
    diff = (m.R2_m - m.R2_f).sort_values().values
    fig, ax = plt.subplots(figsize=(5.2, 2.4), dpi=300)
    ax.bar(range(len(diff)), diff, color=[BLUE if x > 0 else ORANGE for x in diff], width=0.85)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="94 station-direction series, sorted", ylabel="$\\Delta R^2$", xticks=[])
    style(ax)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "station_gain.png")
    print(f"  -> figures/station_gain.png  (median diff {pd.Series(diff).median():+.3f})")


def checks():
    naive = [system(h).query("model == 'Naive (same hour last week)' and phase == 'test'")["R2"].round(4).tolist() for h in H]
    print("  naive-baseline R^2 by horizon (must be identical):", naive)
    legacy = R / "timesfm_system_legacy.csv"
    if legacy.exists():
        print(pd.read_csv(legacy)[["protocol", "target", "R2"]].round(3).to_string(index=False))


def main():
    print("Table: system H=1");        table_system_h1()
    print("Table: horizons");          table_horizons()
    print("Table: stations");          table_stations()
    print("Table: forecast weather");  table_forecast_weather()
    print("Table: ablation");          table_ablation()
    print("Table: gain by volume");    table_volume()
    print("Figures");                  figure_horizon(); figure_station_gain()
    print("Checks");                   checks()


if __name__ == "__main__":
    main()
