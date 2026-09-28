"""Per-station model selection on the validation year (never on the test year)."""

import pandas as pd


def validation_selected(results: pd.DataFrame) -> pd.DataFrame:
    """For each station-target, pick the model with the highest VALIDATION R^2 and return its TEST row.
    `results` has one row per (station_id, target, model, phase) with metric columns."""
    val = results[results["phase"] == "val"]
    test = results[results["phase"] == "test"]
    chosen = val.loc[val.groupby(["station_id", "target"])["R2"].idxmax(), ["station_id", "target", "model"]]
    return chosen.merge(test, on=["station_id", "target", "model"])
