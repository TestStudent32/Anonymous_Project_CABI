"""Zero-shot TimesFM (no fine-tuning, no exogenous inputs) via Hugging Face transformers.

Two evaluation protocols:
  rolling_origin - the fair one used in the paper. For each test hour T and horizon H, the model sees
                   the actual series up to and including T-H (at most `context_length` hours) and its
                   H-step-ahead mean forecast is scored: the same information cutoff as the feature models.
  legacy         - reproduces an earlier, flawed protocol: one recursive forecast of the whole test year
                   from train+validation history (no test-year actual is ever observed).
"""

import numpy as np
import pandas as pd
import torch

import config


class TimesFM:
    def __init__(self, model_id: str = config.TIMESFM_MODEL_ID):
        from transformers import TimesFmModelForPrediction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = TimesFmModelForPrediction.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.context_length = int(getattr(self.model.config, "context_length", 2048))

    def forecast(self, contexts, freq: int = config.TIMESFM_HIGH_FREQ) -> np.ndarray:
        """Mean forecasts (batch, model horizon) for a list of 1-D context arrays, clipped at 0."""
        past = [torch.tensor(c, dtype=self.dtype, device=self.device) for c in contexts]
        freq_t = torch.full((len(contexts),), freq, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(past_values=past, freq=freq_t, return_dict=True)
        return np.maximum(out.mean_predictions.float().cpu().numpy(), 0)


def rolling_origin(tfm: TimesFM, series: pd.Series, test_times: pd.DatetimeIndex, horizons,
                   batch_size: int = 32) -> dict:
    """{H: predictions aligned with test_times}. One forward pass per origin gives every horizon up
    to max(horizons); the H-step forecast from origin T-H is the prediction for hour T."""
    max_h = max(horizons)
    values = series.values.astype(float)
    pos = {t: i for i, t in enumerate(series.index)}
    origins = pd.date_range(test_times.min() - pd.Timedelta(hours=max_h),
                            test_times.max() - pd.Timedelta(hours=1), freq="h")
    by_origin = {}
    for b in range(0, len(origins), batch_size):
        batch = origins[b: b + batch_size]
        ctx = [values[max(0, pos[o] + 1 - tfm.context_length): pos[o] + 1] for o in batch]  # ends at origin o
        for o, f in zip(batch, tfm.forecast(ctx)):
            by_origin[o] = f[:max_h]
    return {h: np.array([by_origin[t - pd.Timedelta(hours=h)][h - 1] for t in test_times]) for h in horizons}


def legacy_recursive(tfm: TimesFM, history: np.ndarray, n_steps: int, freq: int) -> np.ndarray:
    """Forecast n_steps by feeding the model's own predictions back as context."""
    history = history.astype(float)
    preds = []
    while len(preds) < n_steps:
        step = tfm.forecast([history[-tfm.context_length:]], freq=freq)[0][: n_steps - len(preds)]
        preds.extend(step)
        history = np.concatenate([history, step])
    return np.array(preds)
