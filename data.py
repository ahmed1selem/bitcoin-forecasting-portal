"""Helpers for loading and preparing the BTC dataset."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DATE_COLUMN_OPTIONS = ["date", "timestamp", "datetime", "time", "open time", "close time"]
PRICE_COLUMN_OPTIONS = ["close", "open", "high", "low"]
VOLUME_COLUMN_OPTIONS = ["volume", "vol.", "vol", "trade volume"]


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from a candidate list."""
    normalized_names = {column.lower().strip(): column for column in df.columns}

    for option in candidates:
        if option in normalized_names:
            return normalized_names[option]

    return None


def detect_columns(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """Find the date column and any available OHLC price columns."""
    date_col = _find_column(df, DATE_COLUMN_OPTIONS)
    if date_col is None:
        raise ValueError(
            f"Could not find a date column. Expected one of: {DATE_COLUMN_OPTIONS}. "
            f"Found columns: {list(df.columns)}"
        )

    normalized_names = {column.lower().strip(): column for column in df.columns}
    price_cols = []
    for option in PRICE_COLUMN_OPTIONS:
        if option in normalized_names:
            price_cols.append(normalized_names[option])

    if not price_cols:
        raise ValueError(
            f"Could not find any price column. Expected one of: {PRICE_COLUMN_OPTIONS}. "
            f"Found columns: {list(df.columns)}"
        )

    return date_col, price_cols


def load_data(uploaded_file) -> pd.DataFrame:
    """Read a CSV file and clean up the column names."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    return df


def preprocess(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Prepare the dataset for modeling."""
    prepared = df.copy()

    prepared[date_col] = pd.to_datetime(
        prepared[date_col],
        infer_datetime_format=True,
        utc=True,
    )
    prepared[date_col] = prepared[date_col].dt.tz_localize(None)

    prepared = prepared.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    rename_map = {date_col: "ds", price_col: "y"}
    normalized_names = {column.lower().strip(): column for column in prepared.columns}
    for option in VOLUME_COLUMN_OPTIONS:
        if option in normalized_names:
            rename_map[normalized_names[option]] = "volume"
            break

    prepared = prepared.rename(columns=rename_map)

    prepared = prepared.set_index("ds")
    numeric_columns = prepared.select_dtypes(include=[np.number]).columns.tolist()
    prepared = prepared[numeric_columns]
    prepared = prepared.resample("D").last()
    prepared = prepared.ffill()
    prepared = prepared.reset_index()

    if "y" not in prepared.columns:
        raise ValueError(f"Price column '{price_col}' was lost during preprocessing.")

    if start_date is not None:
        prepared = prepared[prepared["ds"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        prepared = prepared[prepared["ds"] <= pd.Timestamp(end_date)]

    prepared = prepared.dropna(subset=["y"])
    prepared = prepared.reset_index(drop=True)

    return prepared


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the features used by the XGBoost model."""
    columns = ["ds", "y"]
    if "volume" in df.columns:
        columns.append("volume")

    features = df[columns].copy()

    features["lag_1"] = features["y"].shift(1)

    if "volume" in features.columns:
        features["volume_lag_1"] = features["volume"].shift(1)

    for window in [7, 30]:
        shifted_price = features["y"].shift(1)
        features[f"rolling_mean_{window}"] = shifted_price.rolling(window).mean()
        features[f"rolling_std_{window}"] = shifted_price.rolling(window).std()

    features["pct_change_1"] = features["y"].shift(1).pct_change(1)

    shifted_price = features["y"].shift(1)

    delta = shifted_price.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    features["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = shifted_price.ewm(span=12, adjust=False).mean()
    ema_26 = shifted_price.ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()

    sma_20 = shifted_price.rolling(window=20).mean()
    std_20 = shifted_price.rolling(window=20).std()
    features["bb_upper"] = sma_20 + (std_20 * 2)
    features["bb_lower"] = sma_20 - (std_20 * 2)

    features["day_of_week"] = features["ds"].dt.dayofweek
    features["month"] = features["ds"].dt.month

    return features


def get_feature_columns() -> List[str]:
    """Return the feature names expected by the XGBoost model."""
    return [
        "lag_1",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_30",
        "pct_change_1",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "day_of_week",
        "month",
        "volume_lag_1",
    ]


def train_test_split_ts(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series without shuffling."""
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df
