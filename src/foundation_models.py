"""Pretrained time-series foundation models, used without fine-tuning, behind one interface.

  TimesFM      google/timesfm-2.0-500m-pytorch  demand history only; mean forecast
  ChronosBolt  amazon/chronos-bolt-base         demand history only; median forecast
  Chronos2     amazon/chronos-2                 demand history + covariates (calendar, weather); median forecast

All are evaluated with the same rolling-origin protocol (`rolling_origin`): for each test hour T and horizon
H, the model sees the actual series up to and including T-H (at most CONTEXT_LENGTH hours) and its H-step
forecast is scored. Covariates, when used, cover the context window and the next max(H) hours; they are
calendar values (known in advance) and weather at those hours (the same perfect-forecast assumption as the
feature models, unless forecast weather is supplied).

`legacy_recursive` reproduces an earlier, flawed TimesFM protocol (one recursive forecast of the whole
test year from train+validation history).
"""

import numpy as np
import pandas as pd
import torch

import config

CONTEXT_LENGTH = 2048  # same maximum context for every model (TimesFM 2.0's limit)


class _Base:
    uses_covariates = False

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32


class TimesFM(_Base):
    name = "TimesFM"

    def __init__(self, model_id: str = config.TIMESFM_MODEL_ID):
        super().__init__()
        from transformers import TimesFmModelForPrediction
        self.model = TimesFmModelForPrediction.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()

    def forecast(self, contexts, horizon, freq: int = config.TIMESFM_HIGH_FREQ, **_) -> np.ndarray:
        """Mean forecasts, shape (batch, >= horizon), clipped at 0."""
        past = [torch.tensor(c, dtype=self.dtype, device=self.device) for c in contexts]
        freq_t = torch.full((len(contexts),), freq, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(past_values=past, freq=freq_t, return_dict=True)
        return np.maximum(out.mean_predictions.float().cpu().numpy(), 0)


class ChronosBolt(_Base):
    name = "Chronos-Bolt"

    def __init__(self, model_id: str = "amazon/chronos-bolt-base"):
        super().__init__()
        from chronos import ChronosBoltPipeline
        self.pipe = ChronosBoltPipeline.from_pretrained(model_id, device_map=self.device, torch_dtype=self.dtype)

    def forecast(self, contexts, horizon, **_) -> np.ndarray:
        """Median (0.5-quantile) forecasts, shape (batch, horizon), clipped at 0."""
        ctx = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        quantiles, _ = self.pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=[0.5])
        return np.maximum(quantiles[..., 0].float().cpu().numpy(), 0)  # (batch, horizon, 1) -> median


class Chronos2(_Base):
    name = "Chronos-2"
    uses_covariates = True

    def __init__(self, model_id: str = "amazon/chronos-2"):
        super().__init__()
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=self.device, torch_dtype=self.dtype)

    def forecast(self, contexts, horizon, past_cov=None, future_cov=None, **_) -> np.ndarray:
        """Median forecasts, shape (batch, horizon), clipped at 0. past_cov / future_cov: one
        {name: array} dict per series (or None for no covariates)."""
        if past_cov is None:
            inputs = [np.asarray(c, dtype=np.float32) for c in contexts]
        else:
            inputs = [{"target": np.asarray(c, dtype=np.float32), "past_covariates": p, "future_covariates": f}
                      for c, p, f in zip(contexts, past_cov, future_cov)]
        q, _ = self.pipe.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=[0.5],
                                           batch_size=len(inputs))
        return np.maximum(np.stack([x[0, :, 0].float().cpu().numpy() for x in q]), 0)


def rolling_origin(model, series: pd.Series, test_times: pd.DatetimeIndex, horizons,
                   covariates: pd.DataFrame = None, batch_size: int = 32,
                   future_covariates: pd.DataFrame = None) -> dict:
    """{H: predictions aligned with test_times}. One forward pass per origin o gives every horizon up to
    max(horizons); the H-step forecast from origin T-H is the prediction for hour T. `covariates` (same
    hourly index as `series`) are passed as past values over the context and future values over the next
    max(horizons) hours. If `future_covariates` is given (same index and columns, e.g. with forecast
    weather), the future window is taken from it while the past window stays observed."""
    max_h = max(horizons)
    values = series.values.astype(float)
    pos = {t: i for i, t in enumerate(series.index)}
    cov = None if covariates is None else {c: covariates[c].values.astype(np.float32) for c in covariates.columns}
    fcov = cov if future_covariates is None else {c: future_covariates[c].values.astype(np.float32) for c in cov}
    # Origins are positions in the elapsed-hour series: the origin for target T at horizon H is H entries
    # (H elapsed hours) before T. Clock arithmetic would be wrong across daylight-saving changes.
    t_pos = np.array([pos[t] for t in test_times])
    origins = range(t_pos.min() - max_h, t_pos.max())
    by_origin = {}
    for b in range(0, len(origins), batch_size):
        batch = origins[b: b + batch_size]
        spans = [(max(0, o + 1 - CONTEXT_LENGTH), o + 1) for o in batch]  # context ends at origin o
        ctx = [values[s:e] for s, e in spans]
        past = fut = None
        if cov is not None:
            past = [{c: v[s:e] for c, v in cov.items()} for s, e in spans]
            # Near the end of the data fewer than max_h future hours exist; pad with the last value.
            # Those steps lie beyond the last test hour and are never scored.
            fut = [{c: np.pad(v[e:e + max_h], (0, max(0, max_h - len(v[e:e + max_h]))), mode="edge")
                    for c, v in fcov.items()} for _, e in spans]
        for o, f in zip(batch, model.forecast(ctx, max_h, past_cov=past, future_cov=fut)):
            by_origin[o] = f[:max_h]
    return {h: np.array([by_origin[p - h][h - 1] for p in t_pos]) for h in horizons}


def legacy_recursive(model: TimesFM, history: np.ndarray, n_steps: int, freq: int) -> np.ndarray:
    """Forecast n_steps by feeding the model's own predictions back as context (flawed legacy protocol)."""
    history = history.astype(float)
    preds = []
    while len(preds) < n_steps:
        step = model.forecast([history[-CONTEXT_LENGTH:]], n_steps, freq=freq)[0][: n_steps - len(preds)]
        preds.extend(step)
        history = np.concatenate([history, step])
    return np.array(preds)
