import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

DEFAULT_ROLLING_WINDOWS = [5, 10, 20]


def compute_historical_rolling_metrics(
    df_historical, feats_to_roll=None, return_feats=None, windows=None, n_workers=None
):
    """
    Pre-compute rolling metrics from historical data.
    Returns a DataFrame with rolling means and standard deviations for each ticker's last date.
    """
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    auction_volume_features = [
        "intraday_open_auction_dollar_volume",
        "intraday_close_auction_dollar_volume",
        "intraday_total_dollar_volume_all",
    ]

    if feats_to_roll is None:
        feats_to_roll = [
            "intraday_last_hour_dollar_volume_ratio",
            "intraday_full_day_volatility",
            "intraday_close_to_vwap",
        ]

    if return_feats is None:
        return_feats = ["intraday_last_hour_return", "intraday_return", "intraday_last_five_minutes_return"]

    # Ensure DataFrame has proper index
    if not isinstance(df_historical.index, pd.MultiIndex):
        df_historical = df_historical.set_index(["ticker", "trade_date"])

    # Process groups in parallel
    groups = [group for _, group in df_historical.groupby(level="ticker")]
    args = [(group, feats_to_roll, return_feats, windows, auction_volume_features) for group in groups]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_historical_group, args))

    # Combine results and get only the last date for each ticker
    df_result = pd.concat(results)
    return df_result.groupby(level="ticker").last()


def process_historical_group(args):
    """
    Process a single ticker's historical data to compute rolling metrics.
    """
    group, feats_to_roll, return_feats, windows, auction_volume_features = args
    results = {}

    # 1) Process auction volume features (only window=5)
    for feat in auction_volume_features:
        rolling_mean_name = f"roll5_mean_{feat}"
        rolling_mean = group[feat].fillna(0).rolling(window=5, min_periods=5).mean()
        results[rolling_mean_name] = rolling_mean

    # 2) Compute rolling means for ratio features
    for w in windows:
        for feat in feats_to_roll:
            rolling_mean_name = f"roll{w}_mean_{feat}"
            rolling_mean = group[feat].rolling(window=w, min_periods=w).mean()
            results[rolling_mean_name] = rolling_mean

    # 3) Compute rolling means and stds for z-score features
    for w in windows:
        for feat in return_feats:
            mean_col = f"roll{w}_mean_{feat}"
            std_col = f"roll{w}_std_{feat}"

            rolling_mean = group[feat].rolling(window=w, min_periods=w).mean()
            rolling_std = group[feat].rolling(window=w, min_periods=w).std()

            results[mean_col] = rolling_mean
            results[std_col] = rolling_std

    return pd.DataFrame(results, index=group.index)


def apply_rolling_metrics(
    df_today, df_historical_metrics, feats_to_roll=None, return_feats=None, windows=[5, 10, 20], epsilon=1e-9
):
    """
    Apply pre-computed rolling metrics to today's data.
    """
    if feats_to_roll is None:
        feats_to_roll = [
            "intraday_last_hour_dollar_volume_ratio",
            "intraday_full_day_volatility",
            "intraday_close_to_vwap",
        ]

    if return_feats is None:
        return_feats = ["intraday_last_hour_return", "intraday_return", "intraday_last_five_minutes_return"]

    auction_volume_features = [
        "intraday_open_auction_dollar_volume",
        "intraday_close_auction_dollar_volume",
        "intraday_total_dollar_volume_all",
    ]

    results = {}

    # Ensure we have ticker as index for easy joining
    df_today = df_today.set_index("ticker") if "ticker" in df_today.columns else df_today

    # 1) Process auction volume ratios (window=5 only)
    for feat in auction_volume_features:
        rolling_mean_name = f"roll5_mean_{feat}"
        if rolling_mean_name in df_historical_metrics.columns:
            results[rolling_mean_name] = df_historical_metrics[rolling_mean_name]

    # 2) Create ratio features
    for w in windows:
        for feat in feats_to_roll:
            ratio_col_name = f"roll{w}_ratio_{feat}"
            rolling_mean_name = f"roll{w}_mean_{feat}"

            if rolling_mean_name in df_historical_metrics.columns:
                results[ratio_col_name] = df_today[feat] / (df_historical_metrics[rolling_mean_name] + epsilon)

    # 3) Create z-score features
    for w in windows:
        for feat in return_feats:
            mean_col = f"roll{w}_mean_{feat}"
            std_col = f"roll{w}_std_{feat}"
            z_col = f"zscore_{w}_{feat}"

            if mean_col in df_historical_metrics.columns and std_col in df_historical_metrics.columns:
                results[z_col] = (df_today[feat] - df_historical_metrics[mean_col]) / df_historical_metrics[
                    std_col
                ].replace(0, np.nan)

    # Combine results with original features
    df_result = pd.concat([df_today, pd.DataFrame(results, index=df_today.index)], axis=1)

    return df_result.reset_index() if "ticker" not in df_result.columns else df_result
