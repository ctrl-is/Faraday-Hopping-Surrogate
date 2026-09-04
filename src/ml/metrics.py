"""
Defines regression metrics for evaluating the models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def compute_regression_metrics(
    y_true: pd.Series | np.ndarray, 
    y_pred: pd.Series | np.ndarray,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    mse = mean_squared_error(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
        )

    return {
        "mae": float(
            mean_absolute_error(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
            )
        ),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(
            r2_score(
                y_true=y_true, y_pred=y_pred, sample_weight=sample_weight,
            )
        ),
        "max_error": float(max_error(y_true=y_true, y_pred=y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
    }


def compare_model_to_baseline(
    model_name: str,
    target_column: str,
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: pd.Series | np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: pd.Series | np.ndarray,
    train_sample_weight: pd.Series | np.ndarray | None = None,
    test_sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, float | str]:
    train_metrics = compute_regression_metrics(
        y_true=y_train_true,
        y_pred=y_train_pred,
        sample_weight=train_sample_weight,
    )
    test_metrics = compute_regression_metrics(
        y_true=y_test_true,
        y_pred=y_test_pred,
        sample_weight=test_sample_weight,
    )

    baseline_value = float(
        np.average(y_train_true, weights=train_sample_weight)
    )
    baseline_test_pred = np.full(shape=len(y_test_true), fill_value=baseline_value)
    baseline_metrics = compute_regression_metrics(
        y_true=y_test_true,
        y_pred=baseline_test_pred,
        sample_weight=test_sample_weight,
    )

    mae_improvement = baseline_metrics["mae"] - test_metrics["mae"]

    if baseline_metrics["mae"] == 0.0:
        mae_improvement_fraction = 0.0
    else:
        mae_improvement_fraction = mae_improvement / baseline_metrics["mae"]

    return {
        "model_name": model_name,
        "target_column": target_column,
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "baseline_test_mae": baseline_metrics["mae"],
        "baseline_test_rmse": baseline_metrics["rmse"],
        "baseline_test_r2": baseline_metrics["r2"],
        "mae_improvement_over_baseline": float(mae_improvement),
        "mae_improvement_fraction": float(mae_improvement_fraction),
    }