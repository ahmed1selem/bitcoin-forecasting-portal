"""Utilities for forecast metrics and model comparison."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a small set of forecast metrics."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    common_length = min(len(actual), len(predicted))
    actual = actual[:common_length]
    predicted = predicted[:common_length]

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    epsilon = 1e-10
    mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100

    if common_length > 1:
        actual_diff = np.sign(np.diff(actual))
        predicted_diff = np.sign(np.diff(predicted))
        mda = np.mean(actual_diff == predicted_diff) * 100
    else:
        mda = np.nan

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "MDA": round(mda, 2) if not np.isnan(mda) else np.nan,
    }


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Turn model results into a comparison table."""
    rows = []

    for model_name, result in results.items():
        metrics = result.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "MAE": metrics.get("MAE", np.nan),
                "RMSE": metrics.get("RMSE", np.nan),
                "MAPE (%)": metrics.get("MAPE", np.nan),
                "MDA (%)": metrics.get("MDA", np.nan),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def select_winner(results: Dict[str, Dict[str, Any]]) -> str:
    """Pick the model with the lowest RMSE."""
    best_name = None
    best_rmse = np.inf

    for model_name, result in results.items():
        rmse = result.get("metrics", {}).get("RMSE", np.inf)
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = model_name

    return best_name
