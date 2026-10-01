"""Step 5 (analysis): system-level feature ablation by horizon.

Feature groups (from src/features.py):
  calendar  season, cyclic hour/day-of-week, holiday/weekend, active hours, COVID flag
  weather   weather at the target hour + 1/3/6/12 h weather lags
  demand    demand lags, rolling means, exponentially weighted means (all horizon-aware)
For each group combination and horizon, four models (linear, polynomial, random forest, LightGBM) are
trained; the one with the best validation R^2 is scored on the test year.

Output: results/ablation_system.csv  (subset, horizon, target, selected model, val R2, test R2, test MAE)
Usage:  python scripts/05_feature_ablation.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_io, features, models  # noqa: E402
from src.metrics import evaluate  # noqa: E402

CAL = features._CALENDAR
WX = features._WEATHER_NOW + features._WEATHER_LAG_COLS
DEMAND = [c for c in features.SYSTEM_FEATURES if c not in CAL and c not in WX]
SUBSETS = {
    "calendar": CAL,
    "calendar+weather": CAL + WX,
    "demand": DEMAND,
    "demand+calendar": [c for c in features.SYSTEM_FEATURES if c not in WX],
    "all": features.SYSTEM_FEATURES,
}
CANDIDATES = ["Linear Regression", "Polynomial Regression", "Random Forest", "LightGBM"]


def main():
    warnings.filterwarnings("ignore")
    holidays = data_io.load_holidays()
    raw = data_io.load_system_data()
    rows = []
    for h in config.HORIZONS:
        train, val, test = data_io.train_val_test_split(features.system_features(raw, holidays, h))
        for subset, cols in SUBSETS.items():
            for target in config.TARGETS:
                best = None
                for name in CANDIDATES:
                    pv, pt = models.FEATURE_MODELS[name](train[cols], val[cols], test[cols], train[target], "system")
                    v = evaluate(val[target], pv)
                    if best is None or v["R2"] > best[1]["R2"]:
                        best = (name, v, evaluate(test[target], pt))
                name, v, t = best
                rows.append({"subset": subset, "horizon": h, "target": target, "model": name,
                             "val_R2": v["R2"], "test_R2": t["R2"], "test_MAE": t["MAE"]})
                print(f"H={h:>2} {subset:<17} {target[:3]}: {name:<22} test R2={t['R2']:.3f}")
    out = config.RESULTS_DIR / "ablation_system.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
