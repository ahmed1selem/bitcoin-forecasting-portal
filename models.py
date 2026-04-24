"""Wrappers around the forecasting models used by the app."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


class ARIMAModel:
    """Small wrapper around pmdarima auto_arima."""

    def __init__(self):
        self.model = None
        self.order = None
        self.train_series = None

    def fit(self, train_df: pd.DataFrame) -> "ARIMAModel":
        """Fit the ARIMA model on the training series."""
        import pmdarima as pm

        self.train_series = train_df["y"].values

        self.model = pm.auto_arima(
            self.train_series,
            start_p=0,
            max_p=5,
            start_q=0,
            max_q=5,
            d=None,
            max_d=2,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            information_criterion="aic",
        )

        self.order = self.model.order
        return self

    def predict(self, horizon: int, confidence: float = 0.95) -> pd.DataFrame:
        """Predict future values and confidence bounds."""
        alpha = 1.0 - confidence
        predictions, confidence_interval = self.model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=alpha,
        )

        return pd.DataFrame(
            {
                "yhat": predictions,
                "yhat_lower": confidence_interval[:, 0],
                "yhat_upper": confidence_interval[:, 1],
            }
        )

    def backtest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Fit on train data, then score on the test period."""
        from evaluation import compute_metrics

        self.fit(train_df)
        horizon = len(test_df)
        forecast = self.predict(horizon, confidence)

        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values
        metrics = compute_metrics(y_true, y_pred)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_lower": forecast["yhat_lower"].values,
            "y_upper": forecast["yhat_upper"].values,
            "dates": test_df["ds"].values,
            "metrics": metrics,
            "params": f"ARIMA{self.order}",
        }

    def get_info(self) -> str:
        """Return a short markdown summary."""
        if self.model is None:
            return "Model not fitted yet."

        return (
            f"**Auto-ARIMA** selected order: **{self.order}**\n\n"
            f"- **p = {self.order[0]}**: Number of autoregressive (AR) terms\n"
            f"- **d = {self.order[1]}**: Differencing order for stationarity\n"
            f"- **q = {self.order[2]}**: Number of moving average (MA) terms\n\n"
            f"AIC: {self.model.aic():.2f}"
        )


class ProphetModel:
    """Wrapper around Prophet with the settings used by the app."""

    def __init__(self):
        self.model = None

    def fit(self, train_df: pd.DataFrame, confidence: float = 0.95) -> "ProphetModel":
        """Train Prophet using the expected ds/y columns."""
        from prophet import Prophet

        self.model = Prophet(
            interval_width=confidence,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )

        prophet_df = train_df[["ds", "y"]].copy()
        self.model.fit(prophet_df)
        return self

    def predict(self, horizon: int, confidence: float = 0.95) -> pd.DataFrame:
        """Create a future forecast frame."""
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
        return result.reset_index(drop=True)

    def backtest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Fit on train data, then score on the test period."""
        from evaluation import compute_metrics

        self.fit(train_df, confidence)
        horizon = len(test_df)
        forecast = self.predict(horizon, confidence)

        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values
        metrics = compute_metrics(y_true, y_pred)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_lower": forecast["yhat_lower"].values,
            "y_upper": forecast["yhat_upper"].values,
            "dates": test_df["ds"].values,
            "metrics": metrics,
            "params": "Prophet (auto-configured)",
        }

    def get_info(self) -> str:
        """Return a short markdown summary."""
        if self.model is None:
            return "Model not fitted yet."

        return (
            "**Facebook Prophet**\n\n"
            "- Weekly seasonality: enabled\n"
            "- Yearly seasonality: enabled\n"
            "- Changepoint prior scale: 0.05 (regularized)\n"
            "- Automatic changepoint detection for trend shifts"
        )


class XGBoostModel:
    """XGBoost model that uses the engineered time-series features."""

    def __init__(self):
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self.feature_cols = None
        self.feature_importances_ = None

    def _get_feature_cols(self):
        """Feature names should stay in sync with data.engineer_features."""
        from data import get_feature_columns

        return get_feature_columns()

    def fit(self, train_feat_df: pd.DataFrame, confidence: float = 0.95) -> "XGBoostModel":
        """Train the point model and the two quantile models."""
        from xgboost import XGBRegressor

        all_feature_cols = self._get_feature_cols()
        self.feature_cols = [column for column in all_feature_cols if column in train_feat_df.columns]
        alpha = (1 - confidence) / 2

        # Lagged features start with NaNs, so we train only on complete rows.
        clean = train_feat_df.dropna(subset=self.feature_cols)
        X = clean[self.feature_cols].values
        y = np.log(clean["y"] / clean["lag_1"]).values

        self.model_median = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        self.model_median.fit(X, y, eval_set=[(X, y)], verbose=False)

        self.model_lower = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        self.model_lower.fit(X, y, eval_set=[(X, y)], verbose=False)

        self.model_upper = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=1 - alpha,
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        self.model_upper.fit(X, y, eval_set=[(X, y)], verbose=False)

        self.feature_importances_ = dict(
            zip(self.feature_cols, self.model_median.feature_importances_)
        )

        return self

    def _predict_one_step(self, features: np.ndarray) -> Tuple[float, float, float]:
        """Predict one step ahead with all three XGBoost models."""
        X = features.reshape(1, -1)
        return (
            float(self.model_median.predict(X)[0]),
            float(self.model_lower.predict(X)[0]),
            float(self.model_upper.predict(X)[0]),
        )

    def predict_recursive(self, base_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast multiple future steps by feeding each prediction back in."""
        from data import engineer_features

        history = base_df.copy()
        predictions = []

        for _ in range(horizon):
            features_df = engineer_features(history)
            last_row = features_df.iloc[-1]
            feature_values = last_row[self.feature_cols].values.astype(float)

            if np.any(np.isnan(feature_values)):
                if predictions:
                    pred_med = predictions[-1]["yhat"]
                else:
                    pred_med = history["y"].iloc[-1]
                pred_lo = pred_med * 0.95
                pred_hi = pred_med * 1.05
            else:
                diff_med, diff_lo, diff_hi = self._predict_one_step(feature_values)
                last_price = history["y"].iloc[-1]
                pred_med = last_price * np.exp(diff_med)
                pred_lo = last_price * np.exp(diff_lo)
                pred_hi = last_price * np.exp(diff_hi)

            last_date = history["ds"].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)

            predictions.append(
                {
                    "ds": next_date,
                    "yhat": pred_med,
                    "yhat_lower": pred_lo,
                    "yhat_upper": pred_hi,
                }
            )

            new_row_data = {"ds": [next_date], "y": [pred_med]}
            for column in history.columns:
                if column not in ["ds", "y"]:
                    new_row_data[column] = [history[column].iloc[-1]]

            new_row = pd.DataFrame(new_row_data)
            history = pd.concat([history, new_row], ignore_index=True)

        return pd.DataFrame(predictions)

    def backtest(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_feat_df: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Run a one-step walk-forward style backtest on the test window."""
        from data import engineer_features
        from evaluation import compute_metrics

        self.fit(train_feat_df, confidence)

        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_feat = engineer_features(full_df)
        test_feat_df = full_feat.iloc[-len(test_df):].copy()

        clean = test_feat_df.dropna(subset=self.feature_cols + ["lag_1"])
        if len(clean) == 0:
            raise ValueError("Test set is too small to construct valid features.")

        X_test = clean[self.feature_cols].values

        pred_log_ret_med = self.model_median.predict(X_test)
        pred_log_ret_lo = self.model_lower.predict(X_test)
        pred_log_ret_hi = self.model_upper.predict(X_test)

        last_prices = clean["lag_1"].values
        y_pred = last_prices * np.exp(pred_log_ret_med)
        y_lower = last_prices * np.exp(pred_log_ret_lo)
        y_upper = last_prices * np.exp(pred_log_ret_hi)

        y_true = clean["y"].values
        metrics = compute_metrics(y_true, y_pred)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "dates": clean["ds"].values,
            "metrics": metrics,
            "params": "XGBoost + Log Returns (1-Step Walk-Forward)",
        }

    def get_info(self) -> str:
        """Return a short markdown summary."""
        if self.model_median is None:
            return "Model not fitted yet."

        sorted_features = sorted(
            self.feature_importances_.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        feature_lines = "\n".join(
            f"  - `{name}`: {importance:.4f}" for name, importance in sorted_features
        )

        return (
            "**XGBoost with Engineered Features**\n\n"
            "**Research basis:** Chen et al. (2020) — tree-based ensembles with\n"
            "temporal features outperform ARIMA on BTC by capturing non-linear\n"
            "regime changes.\n\n"
            "**Configuration:**\n"
            "- 500 estimators, max_depth=5, lr=0.05\n"
            "- Early stopping (30 rounds)\n"
            "- Quantile regression for confidence intervals\n"
            "- Recursive multi-step forecasting\n\n"
            f"**Feature Importances:**\n{feature_lines}"
        )
