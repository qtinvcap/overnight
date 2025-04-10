import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# SECTION 2: HELPER UTILITIES
# =============================================================================


def calc_return_for_window(df_window):
    """
    Given a subset of the DataFrame for a time window,
    compute the return = (last_close / first_open - 1).
    Returns a Series of returns indexed by 'ticker'.
    """
    if df_window.empty:
        return pd.Series(dtype="float64")
    grouped_open = df_window.groupby("ticker")["open"].first()
    grouped_close = df_window.groupby("ticker")["close"].last()
    return (grouped_close / grouped_open - 1).fillna(0)


def get_intraday_last_close_price_before_1555(df):
    df_last_close_before_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()]
    return df_last_close_before_1555.groupby("ticker")["close"].last()


# =============================================================================
# SECTION 2.5: NEW FEATURE (EXISTING)
# =============================================================================


def get_price_acceleration(df):
    """
    Compute the average second derivative (acceleration) of the close price
    using four different intraday windows:
      - Last 5 minutes (15:50–15:55)
      - Last 30 minutes (15:25–15:55)
      - Last hour (15:45–15:55)
      - Second half-day (13:00–15:55)
    Returns a tuple of 4 Series corresponding to each time window.
    """

    df_last30_min = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:25").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:45").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_second_half_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("13:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]

    def compute_acceleration(group):
        """
        Compute price acceleration using quadratic regression.
        Returns 0 if fewer than 3 data points to avoid NaN.
        """
        if len(group) < 3:
            return 0
        time = (group["window_start"] - group["window_start"].min()).dt.total_seconds()
        price = group["close"]
        try:
            coeffs = np.polyfit(time, price, 2)  # Fit quadratic: a*t^2 + b*t + c
            acceleration = 2 * coeffs[0]  # Acceleration is 2*a
        except np.linalg.LinAlgError:
            acceleration = 0
        return acceleration

    if df_last30_min.empty:
        tickers = df["ticker"].unique()
        acceleration_feature_30min = pd.Series(np.nan, index=tickers, name="intraday_last30_price_acceleration")
    else:
        acceleration_feature_30min = df_last30_min.groupby("ticker").apply(compute_acceleration)
        acceleration_feature_30min.name = "intraday_last30_price_acceleration"

    if df_last_hour.empty:
        tickers = df["ticker"].unique()
        acceleration_feature_1hour = pd.Series(np.nan, index=tickers, name="intraday_last_hour_price_acceleration")
    else:
        acceleration_feature_1hour = df_last_hour.groupby("ticker").apply(compute_acceleration)
        acceleration_feature_1hour.name = "intraday_last_hour_price_acceleration"

    if df_second_half_day.empty:
        tickers = df["ticker"].unique()
        acceleration_feature_second_half_day = pd.Series(
            np.nan, index=tickers, name="intraday_second_half_day_price_acceleration"
        )
    else:
        acceleration_feature_second_half_day = df_second_half_day.groupby("ticker").apply(compute_acceleration)
        acceleration_feature_second_half_day.name = "intraday_second_half_day_price_acceleration"

    return (
        acceleration_feature_30min,
        acceleration_feature_1hour,
        acceleration_feature_second_half_day,
    )


# =============================================================================
# SECTION 3.1: VOLUME FEATURES
# =============================================================================


def get_open_close_auction_dollar_volume(df):
    df_open_auction = df[df["window_start"].dt.time == pd.to_datetime("09:30").time()]
    df_close_auction = df[df["window_start"].dt.time == pd.to_datetime("16:00").time()]
    df_open_auction_dollar_volume = (
        (df_open_auction["volume"] * df_open_auction["close"]).groupby(df_open_auction["ticker"]).sum()
    )
    df_close_auction_dollar_volume = (
        (df_close_auction["volume"] * df_close_auction["close"]).groupby(df_close_auction["ticker"]).sum()
    )
    return df_open_auction_dollar_volume, df_close_auction_dollar_volume


def get_intraday_volume_features(df):
    # Time slicing for various windows.
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_quarter_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:45").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_first_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("10:30").time())
    ]
    last_hour_dollar_volume = (df_last_hour["volume"] * df_last_hour["close"]).groupby(df_last_hour["ticker"]).sum()
    last_half_hour_dollar_volume = (
        (df_last_half_hour["volume"] * df_last_half_hour["close"]).groupby(df_last_half_hour["ticker"]).sum()
    )
    last_quarter_hour_dollar_volume = (
        (df_last_quarter_hour["volume"] * df_last_quarter_hour["close"]).groupby(df_last_quarter_hour["ticker"]).sum()
    )
    first_hour_dollar_volume = (df_first_hour["volume"] * df_first_hour["close"]).groupby(df_first_hour["ticker"]).sum()

    df_until_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()]
    total_dollar_volume_until_1555 = (
        (df_until_1555["volume"] * df_until_1555["close"]).groupby(df_until_1555["ticker"]).sum()
    )
    total_dollar_volume_all = (df["volume"] * df["close"]).groupby(df["ticker"]).sum()

    last_hour_dollar_volume_ratio = (last_hour_dollar_volume / total_dollar_volume_until_1555).fillna(0)
    last_half_hour_dollar_volume_ratio = (last_half_hour_dollar_volume / total_dollar_volume_until_1555).fillna(0)
    last_quarter_hour_dollar_volume_ratio = (last_quarter_hour_dollar_volume / total_dollar_volume_until_1555).fillna(0)
    first_hour_dollar_volume_ratio = (first_hour_dollar_volume / total_dollar_volume_until_1555).fillna(0)
    return (
        last_hour_dollar_volume_ratio,
        last_half_hour_dollar_volume_ratio,
        last_quarter_hour_dollar_volume_ratio,
        first_hour_dollar_volume_ratio,
        total_dollar_volume_until_1555,
        total_dollar_volume_all,
    )


# =============================================================================
# SECTION 3.2: VOLATILITY FEATURES
# =============================================================================


def get_intraday_volatility_features(df):
    df_full_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_quarter_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:45").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_hour_volatility = df_last_hour.groupby("ticker")["open"].std()
    last_half_hour_volatility = df_last_half_hour.groupby("ticker")["open"].std()
    last_quarter_hour_volatility = df_last_quarter_hour.groupby("ticker")["open"].std()
    full_day_volatility = df_full_day.groupby("ticker")["open"].std()
    return last_hour_volatility, last_half_hour_volatility, last_quarter_hour_volatility, full_day_volatility


# =============================================================================
# SECTION 3.3: PRICE FEATURES
# =============================================================================


def get_intraday_price_features(df):
    # A) Daily Range and Intraday Return
    df_intraday = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    grouped_high = df_intraday.groupby("ticker")["high"].max()
    grouped_low = df_intraday.groupby("ticker")["low"].min()
    daily_range = (grouped_high - grouped_low).fillna(0) / grouped_low.replace(0, pd.NA)

    first_low = df_intraday.groupby("ticker")["low"].first()
    intraday_range = ((grouped_high - grouped_low) / first_low).fillna(0)

    mean_intraday_range = intraday_range.mean()
    median_intraday_range = intraday_range.median()

    df_930 = df[df["window_start"].dt.time >= pd.to_datetime("09:30").time()].groupby("ticker")
    df_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()].groupby("ticker")
    open_930 = df_930["open"].first()
    close_1555 = df_1555["close"].last()
    intraday_return = ((close_1555 / open_930) - 1).fillna(0)
    # B) Partial-window returns
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_hour_return = calc_return_for_window(df_last_hour)
    df_last_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_half_hour_return = calc_return_for_window(df_last_half_hour)
    df_last_quarter_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:40").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_quarter_hour_return = calc_return_for_window(df_last_quarter_hour)
    df_last_ten_minutes = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:45").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_ten_minutes_return = calc_return_for_window(df_last_ten_minutes)
    df_last_five_minutes = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:50").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_five_minutes_return = calc_return_for_window(df_last_five_minutes)
    df_first_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("10:30").time())
    ]
    first_hour_return = calc_return_for_window(df_first_hour)
    df_first_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("10:00").time())
    ]
    first_half_hour_return = calc_return_for_window(df_first_half_hour)
    df_first_five_minutes = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("09:35").time())
    ]
    first_five_minutes_return = calc_return_for_window(df_first_five_minutes)
    df_first_half_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("13:00").time())
    ]
    first_half_day_return = calc_return_for_window(df_first_half_day)
    df_second_half_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("13:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    second_half_day_return = calc_return_for_window(df_second_half_day)
    df_mid_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("11:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("14:00").time())
    ]
    df_pre_market = df[df["window_start"].dt.time < pd.to_datetime("09:30").time()]
    pre_market_return = calc_return_for_window(df_pre_market)

    mid_day_return = calc_return_for_window(df_mid_day)
    return (
        daily_range.fillna(0),
        mean_intraday_range,
        median_intraday_range,
        intraday_return,
        last_hour_return,
        last_half_hour_return,
        last_quarter_hour_return,
        last_ten_minutes_return,
        last_five_minutes_return,
        first_hour_return,
        first_half_hour_return,
        first_five_minutes_return,
        first_half_day_return,
        second_half_day_return,
        mid_day_return,
        pre_market_return,
    )


# =============================================================================
# SECTION 3.4: TRANSACTION FEATURES
# =============================================================================


def get_intraday_transaction_features(df):
    df["avg_txn_size"] = df["volume"] / df["transactions"].replace(0, pd.NA)
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_quarter_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:45").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_until_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()]

    df_full_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    total_txn_until_1555 = df_until_1555.groupby("ticker")["transactions"].sum()
    last_hour_transactions = df_last_hour.groupby("ticker")["transactions"].sum()
    last_half_hour_transactions = df_last_half_hour.groupby("ticker")["transactions"].sum()
    last_quarter_hour_transactions = df_last_quarter_hour.groupby("ticker")["transactions"].sum()
    last_hour_transactions_ratio = (last_hour_transactions / total_txn_until_1555).fillna(0)
    last_half_hour_transactions_ratio = (last_half_hour_transactions / total_txn_until_1555).fillna(0)
    last_quarter_hour_transactions_ratio = (last_quarter_hour_transactions / total_txn_until_1555).fillna(0)
    avg_txn_size_full_day = df_full_day.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_hour = df_last_hour.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_half_hour = df_last_half_hour.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_quarter_hour = df_last_quarter_hour.groupby("ticker")["avg_txn_size"].mean()
    txn_volatility_full_day = df_full_day.groupby("ticker")["transactions"].std()
    avg_txn_size_volatility_full_day = df_full_day.groupby("ticker")["avg_txn_size"].std()
    return {
        "intraday_last_hour_transactions_ratio": last_hour_transactions_ratio,
        "intraday_last_half_hour_transactions_ratio": last_half_hour_transactions_ratio,
        "intraday_last_quarter_hour_transactions_ratio": last_quarter_hour_transactions_ratio,
        "intraday_avg_txn_size_full_day": avg_txn_size_full_day,
        "intraday_avg_txn_size_last_hour": avg_txn_size_last_hour,
        "intraday_avg_txn_size_last_half_hour": avg_txn_size_last_half_hour,
        "intraday_avg_txn_size_last_quarter_hour": avg_txn_size_last_quarter_hour,
        "intraday_txn_volatility_full_day": txn_volatility_full_day,
        "intraday_avg_txn_size_volatility_full_day": avg_txn_size_volatility_full_day,
    }


# =============================================================================
# SECTION 3.5: VWAP FEATURES
# =============================================================================


def get_intraday_vwap_features(df):
    df_full_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ].copy()
    df_full_day["typical_price"] = (df_full_day["high"] + df_full_day["low"] + df_full_day["close"]) / 3.0
    df_full_day["dollar_volume"] = df_full_day["typical_price"] * df_full_day["volume"]
    grp = df_full_day.groupby("ticker")
    sum_dollar_volume = grp["dollar_volume"].sum()
    sum_volume = grp["volume"].sum()
    vwap_full_day = sum_dollar_volume / sum_volume

    open_930 = df_full_day.groupby("ticker")["open"].first()
    ratio_vwap_to_open = open_930 / vwap_full_day
    df_last_hour = df_full_day[df_full_day["window_start"].dt.time >= pd.to_datetime("15:00").time()]
    grp_last_hour = df_last_hour.groupby("ticker")
    sum_dv_hour = grp_last_hour["dollar_volume"].sum()
    sum_vol_hour = grp_last_hour["volume"].sum()
    vwap_last_hour = sum_dv_hour / sum_vol_hour
    open_last_hour = df_last_hour.groupby("ticker")["open"].first()
    ratio_vwap_to_open_last_hour = open_last_hour / vwap_last_hour
    return {
        "intraday_ratio_vwap_full_day_to_open": ratio_vwap_to_open.fillna(0),
        "intraday_ratio_vwap_last_hour_to_open": ratio_vwap_to_open_last_hour.fillna(0),
        "intraday_dollar_volume_full_day": sum_dollar_volume.fillna(0),
        "intraday_dollar_volume_last_hour": sum_dv_hour.fillna(0),
    }


# =============================================================================
# SECTION 3.6: ADDITIONAL MOMENTUM FEATURES
# =============================================================================


def get_additional_momentum_features(df):
    df_intraday = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ].copy()

    df_intraday_including_pre_market = df_intraday[
        (df_intraday["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    df_last_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:00").time()]
    df_last_half_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:30").time()]
    df_last_quarter_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:45").time()]
    last_hour_bar_count = df_last_hour.groupby("ticker").size()
    last_half_hour_bar_count = df_last_half_hour.groupby("ticker").size()
    last_quarter_hour_bar_count = df_last_quarter_hour.groupby("ticker").size()

    def slope_calc(group):
        if group.empty:
            return pd.Series({"intra_slope": 0, "bar_count": 0})
        open_price = group.iloc[0]["open"]
        close_price = group.iloc[-1]["close"]
        bar_count = len(group)
        slope_val = (close_price - open_price) / bar_count if bar_count > 0 else 0
        return pd.Series({"intra_slope": slope_val, "bar_count": bar_count})

    slope_df = df_intraday.groupby("ticker").apply(slope_calc)[["intra_slope", "bar_count"]]
    df_intraday["typical_price"] = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]) / 3.0
    df_intraday["dollar_volume"] = df_intraday["typical_price"] * df_intraday["volume"]
    grp = df_intraday.groupby("ticker")
    sum_dv = grp["dollar_volume"].sum()
    sum_vol = grp["volume"].sum()
    vwap = sum_dv / sum_vol
    final_close = grp["close"].last()
    price_to_vwap = ((final_close - vwap) / vwap).fillna(0)
    vol = grp["close"].std()
    price_to_vwap_zscore = ((final_close - vwap) / (vol + 1e-9)).fillna(0)  # NEW

    # SAME AS ABOVE BUT WITH PRE-MARKET INCLUDED

    df_intraday_including_pre_market["typical_price"] = (
        df_intraday_including_pre_market["high"]
        + df_intraday_including_pre_market["low"]
        + df_intraday_including_pre_market["close"]
    ) / 3.0
    df_intraday_including_pre_market["dollar_volume"] = (
        df_intraday_including_pre_market["typical_price"] * df_intraday_including_pre_market["volume"]
    )
    grp_all_pre_market = df_intraday_including_pre_market.groupby("ticker")
    sum_dv = grp_all_pre_market["dollar_volume"].sum()
    sum_vol = grp_all_pre_market["volume"].sum()
    vwap = sum_dv / sum_vol
    final_close = grp_all_pre_market["close"].last()
    price_to_vwap_all_pre_market = ((final_close - vwap) / vwap).fillna(0)
    vol = grp_all_pre_market["close"].std()
    price_to_vwap_zscore_all_pre_market = ((final_close - vwap) / (vol + 1e-9)).fillna(0)
    return {
        "intraday_intra_slope": slope_df["intra_slope"],
        "intraday_bar_count": slope_df["bar_count"],
        "intraday_close_to_vwap": price_to_vwap,
        "intraday_close_to_vwap_zscore": price_to_vwap_zscore,
        "intraday_last_hour_bar_count": last_hour_bar_count,
        "intraday_last_half_hour_bar_count": last_half_hour_bar_count,
        "intraday_last_quarter_hour_bar_count": last_quarter_hour_bar_count,
        "intraday_pre_market_included_close_to_vwap": price_to_vwap_all_pre_market,
        "intraday_pre_market_included_close_to_vwap_zscore": price_to_vwap_zscore_all_pre_market,
    }


# =============================================================================
# SECTION 3.7: HELPER FOR BAR RATIO
# =============================================================================


def ratio_first_open_to_last_close_bar_before_close(df):
    df_before_1555 = df[df["window_start"].dt.time < pd.to_datetime("15:55").time()]
    df_at_0930 = df[df["window_start"].dt.time >= pd.to_datetime("09:30").time()]
    last_bar_before_1555 = df_before_1555.groupby("ticker")["close"].last()
    first_bar_after_0930 = df_at_0930.groupby("ticker")["open"].first()
    return last_bar_before_1555 / first_bar_after_0930


# =============================================================================
# SECTION 3.8: ADDITIONAL TIME WINDOW FEATURES (New)
# =============================================================================


def precompute_time_windows(df):
    """
    Precompute filtered DataFrames for commonly used intraday time windows.
    Returns a dictionary with keys: 'full_day', 'morning', 'midday', and 'last_hour'.
    """
    windows = {
        "full_day": df[
            (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
            & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
        ],
        "morning": df[
            (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
            & (df["window_start"].dt.time <= pd.to_datetime("11:00").time())
        ],
        "midday": df[
            (df["window_start"].dt.time >= pd.to_datetime("11:30").time())
            & (df["window_start"].dt.time <= pd.to_datetime("14:00").time())
        ],
        "last_hour": df[
            (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
            & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
        ],
    }
    return windows


def get_morning_features_from_windows(windows):
    # Reset the index so that 'ticker' becomes a column.
    df_morning = windows["morning"].copy()
    df_full_day = windows["full_day"].copy()

    # Calculate high_low_ratio explicitly
    df_morning["high_low_ratio"] = (df_morning["high"] - df_morning["low"]) / df_morning["low"]

    # Use the 09:30 open from full_day (filtered for exactly 09:30)
    df_0930 = df_full_day[df_full_day["window_start"].dt.time == pd.to_datetime("09:30").time()]
    open_0930 = df_0930.groupby("ticker")["open"].first()

    # Extract the 11:00 close from the morning DataFrame.
    df_1100 = df_morning[df_morning["window_start"].dt.time == pd.to_datetime("11:00").time()]
    close_1100 = df_1100.groupby("ticker")["close"].last()

    morning_return = (close_1100 / open_0930 - 1).fillna(0)
    morning_dollar_volume = (df_morning["volume"] * df_morning["close"]).groupby(df_morning["ticker"]).sum()

    morning_avg_hl_ratio = df_morning.groupby(df_morning["ticker"])["high_low_ratio"].mean()
    morning_std_hl_ratio = df_morning.groupby(df_morning["ticker"])["high_low_ratio"].std()

    features = pd.DataFrame(
        {
            "intraday_morning_return": morning_return,
            "intraday_morning_dollar_volume": morning_dollar_volume,
            "intraday_morning_avg_hl_ratio": morning_avg_hl_ratio,
            "intraday_morning_hl_ratio_vol": morning_std_hl_ratio,
        }
    )
    return features


def get_midday_features_from_windows(windows):
    df_midday = windows["midday"]
    midday_high = df_midday.groupby("ticker")["high"].max()
    midday_low = df_midday.groupby("ticker")["low"].min()
    midday_range = ((midday_high - midday_low) / midday_low.replace(0, np.nan)).fillna(0)
    df_midday = df_midday.copy()
    df_midday["typical_price"] = (df_midday["high"] + df_midday["low"] + df_midday["close"]) / 3.0
    df_midday["dollar_volume"] = df_midday["typical_price"] * df_midday["volume"]
    grp_midday = df_midday.groupby("ticker")
    vwap_midday = grp_midday["dollar_volume"].sum() / grp_midday["volume"].sum()
    midday_close = grp_midday["close"].last()
    midday_price_to_vwap = ((midday_close - vwap_midday) / vwap_midday).fillna(0)
    features = pd.DataFrame(
        {
            "intraday_midday_range": midday_range,
            "intraday_midday_price_to_vwap_deviation": midday_price_to_vwap,
        }
    )
    return features


def get_late_session_features_from_windows(windows):
    df_last_hour = windows["last_hour"]
    late_return = calc_return_for_window(df_last_hour)
    morning_feats = get_morning_features_from_windows(windows)
    late_to_morning_return_ratio = late_return / (morning_feats["intraday_morning_return"] + 1e-9)
    features = pd.DataFrame(
        {
            "intraday_late_return": late_return,
            "intraday_late_to_morning_return_ratio": late_to_morning_return_ratio,
        }
    )
    return features


# =============================================================================
# SECTION 4: FINAL ASSEMBLER
# =============================================================================


def assemble_all_intraday_features(df, etf_data):
    """
    For a given single-day DataFrame, compute all intraday features:
      - volume, volatility, price, transaction, VWAP, momentum, and the new time-window features.
    Combine them into one DataFrame (one row per ticker).
    """
    last_close_before_1555 = get_intraday_last_close_price_before_1555(df)
    open_auction_dollar_volume, close_auction_dollar_volume = get_open_close_auction_dollar_volume(df)
    (last_30min_price_acceleration, last_hour_price_acceleration, second_half_day_price_acceleration) = (
        get_price_acceleration(df)
    )

    (
        last_hour_dollar_volume_ratio,
        last_half_hour_dollar_volume_ratio,
        last_quarter_hour_dollar_volume_ratio,
        first_hour_dollar_volume_ratio,
        total_dollar_volume_until_1555,
        total_dollar_volume_all,
    ) = get_intraday_volume_features(df)

    (last_hour_volatility, last_half_hour_volatility, last_quarter_hour_volatility, full_day_volatility) = (
        get_intraday_volatility_features(df)
    )

    (
        daily_range,
        mean_intraday_range,
        median_intraday_range,
        intraday_return,
        last_hour_return,
        last_half_hour_return,
        last_quarter_hour_return,
        last_ten_minutes_return,
        last_five_minutes_return,
        first_hour_return,
        first_half_hour_return,
        first_five_minutes_return,
        first_half_day_return,
        second_half_day_return,
        mid_day_return,
        pre_market_return,
    ) = get_intraday_price_features(df)

    txn_dict = get_intraday_transaction_features(df)
    vwap_dict = get_intraday_vwap_features(df)
    momentum_dict = get_additional_momentum_features(df)
    ratio_first_to_last_bar_at_1550 = ratio_first_open_to_last_close_bar_before_close(df)

    # Precompute common time windows to avoid duplicate filtering
    precomputed_windows = precompute_time_windows(df)
    morning_feats = get_morning_features_from_windows(precomputed_windows)
    midday_feats = get_midday_features_from_windows(precomputed_windows)
    late_feats = get_late_session_features_from_windows(precomputed_windows)

    # Combine all features into a dictionary of Series
    data = {
        "intraday_open_auction_dollar_volume": open_auction_dollar_volume,
        "intraday_close_auction_dollar_volume": close_auction_dollar_volume,
        # Price acceleration
        "intraday_last_30min_price_acceleration": last_30min_price_acceleration,
        "intraday_last_hour_price_acceleration": last_hour_price_acceleration,
        "intraday_second_half_day_price_acceleration": second_half_day_price_acceleration,
        # Volume features
        "intraday_last_hour_dollar_volume_ratio": last_hour_dollar_volume_ratio,
        "intraday_last_half_hour_dollar_volume_ratio": last_half_hour_dollar_volume_ratio,
        "intraday_last_quarter_hour_dollar_volume_ratio": last_quarter_hour_dollar_volume_ratio,
        "intraday_first_hour_dollar_volume_ratio": first_hour_dollar_volume_ratio,
        "intraday_total_dollar_volume_until_1555": total_dollar_volume_until_1555,
        "intraday_total_dollar_volume_all": total_dollar_volume_all,
        # Volatility features
        "intraday_last_hour_volatility": last_hour_volatility,
        "intraday_last_half_hour_volatility": last_half_hour_volatility,
        "intraday_last_quarter_hour_volatility": last_quarter_hour_volatility,
        "intraday_full_day_volatility": full_day_volatility,
        # Price features
        "intraday_daily_range": daily_range,
        "intraday_mean_intraday_range": mean_intraday_range,
        "intraday_median_intraday_range": median_intraday_range,
        "intraday_return": intraday_return,
        "intraday_last_hour_return": last_hour_return,
        "intraday_last_half_hour_return": last_half_hour_return,
        "intraday_last_quarter_hour_return": last_quarter_hour_return,
        "intraday_last_ten_minutes_return": last_ten_minutes_return,
        "intraday_last_five_minutes_return": last_five_minutes_return,
        "intraday_first_hour_return": first_hour_return,
        "intraday_first_half_hour_return": first_half_hour_return,
        "intraday_first_five_minutes_return": first_five_minutes_return,
        "intraday_first_half_day_return": first_half_day_return,
        "intraday_second_half_day_return": second_half_day_return,
        "intraday_mid_day_return": mid_day_return,
        "intraday_pre_market_return": pre_market_return,
        "intraday_ratio_first_open_to_last_close_bar": ratio_first_to_last_bar_at_1550,
        # last close before 1555
        "intraday_last_close_before_1555": last_close_before_1555,
    }
    data.update(txn_dict)
    data.update(vwap_dict)
    data.update(momentum_dict)
    # Add the new time-window features
    data.update(morning_feats.to_dict(orient="series"))
    data.update(midday_feats.to_dict(orient="series"))
    data.update(late_feats.to_dict(orient="series"))
    for etf_ticker, df_etf in etf_data.items():
        etf_features = get_etf_comparative_features(df, df_etf, etf_ticker)
        data.update(etf_features)

    # Add raw ETF features
    for etf_ticker, df_etf in etf_data.items():
        # Acceleration
        accel_etf = get_price_acceleration(df_etf)
        data[f"{etf_ticker}_intraday_last_30min_price_acceleration"] = (
            accel_etf[0].iloc[0] if not accel_etf[0].empty else np.nan
        )
        data[f"{etf_ticker}_intraday_last_hour_price_acceleration"] = (
            accel_etf[1].iloc[0] if not accel_etf[1].empty else np.nan
        )
        data[f"{etf_ticker}_intraday_second_half_day_price_acceleration"] = (
            accel_etf[2].iloc[0] if not accel_etf[2].empty else np.nan
        )

        # Returns
        etf_return_full_day = calc_return_for_window(df_etf)
        etf_return_last_hour = calc_return_for_window(
            df_etf[
                (df_etf["window_start"].dt.time >= pd.to_datetime("15:00").time())
                & (df_etf["window_start"].dt.time <= pd.to_datetime("15:55").time())
            ]
        )
        etf_return_last_ten_minutes = calc_return_for_window(
            df_etf[
                (df_etf["window_start"].dt.time >= pd.to_datetime("15:45").time())
                & (df_etf["window_start"].dt.time <= pd.to_datetime("15:55").time())
            ]
        )
        etf_return_last_five_minutes = calc_return_for_window(
            df_etf[
                (df_etf["window_start"].dt.time >= pd.to_datetime("15:50").time())
                & (df_etf["window_start"].dt.time <= pd.to_datetime("15:55").time())
            ]
        )
        data[f"{etf_ticker}_intraday_return"] = etf_return_full_day.iloc[0] if not etf_return_full_day.empty else np.nan
        data[f"{etf_ticker}_intraday_last_hour_return"] = (
            etf_return_last_hour.iloc[0] if not etf_return_last_hour.empty else np.nan
        )
        data[f"{etf_ticker}_intraday_last_ten_minutes_return"] = (
            etf_return_last_ten_minutes.iloc[0] if not etf_return_last_ten_minutes.empty else np.nan
        )
        data[f"{etf_ticker}_intraday_last_five_minutes_return"] = (
            etf_return_last_five_minutes.iloc[0] if not etf_return_last_five_minutes.empty else np.nan
        )

        # Volume
        etf_volume_full_day = (df_etf["volume"] * df_etf["close"]).sum()
        data[f"{etf_ticker}_intraday_total_dollar_volume"] = etf_volume_full_day

        # Volatility
        etf_volatility_full_day = df_etf["open"].std()
        data[f"{etf_ticker}_intraday_full_day_volatility"] = etf_volatility_full_day
        # VWAP
        vwap_etf = get_intraday_vwap_features(df_etf)
        data[f"{etf_ticker}_intraday_ratio_vwap_full_day_to_open"] = (
            vwap_etf["intraday_ratio_vwap_full_day_to_open"].iloc[0]
            if not vwap_etf["intraday_ratio_vwap_full_day_to_open"].empty
            else np.nan
        )

        # Add close-to-VWAP
        momentum_etf = get_additional_momentum_features(df_etf)
        data[f"{etf_ticker}_intraday_close_to_vwap"] = (
            momentum_etf["intraday_close_to_vwap"].iloc[0]
            if not momentum_etf["intraday_close_to_vwap"].empty
            else np.nan
        )

        # Slope
        data[f"{etf_ticker}_intraday_slope"] = (
            momentum_etf["intraday_intra_slope"].iloc[0] if not momentum_etf["intraday_intra_slope"].empty else np.nan
        )

    # --- NEW: Ensure all values in our data dict are 1-dimensional.
    for key, value in data.items():
        if hasattr(value, "ndim") and value.ndim == 2:
            # If it's a one–column DataFrame, convert to Series.
            if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
                data[key] = value.iloc[:, 0]
            else:
                raise ValueError(f"Feature {key} is 2-dimensional: shape {value.shape}")

    combined_df = pd.DataFrame(data)
    combined_df.index.name = "ticker"
    return combined_df


# =============================================================================
# SECTION 5: ETF RELATIVE RETURN
# =============================================================================


def compute_relative_return(df_stock, df_etf, start_time, end_time):
    window_df_stock = df_stock[
        (df_stock["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_stock["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    window_df_etf = df_etf[
        (df_etf["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_etf["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    stock_return = calc_return_for_window(window_df_stock)
    etf_return = calc_return_for_window(window_df_etf)
    return stock_return - etf_return.reindex(stock_return.index).fillna(0)


def compute_volume_ratio(df_stock, df_etf, start_time, end_time):
    window_df_stock = df_stock[
        (df_stock["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_stock["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    window_df_etf = df_etf[
        (df_etf["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_etf["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    # Calculate dollar volume while preserving the ticker column
    stock_dollar_volume = window_df_stock.copy()
    stock_dollar_volume["dollar_volume"] = stock_dollar_volume["volume"] * stock_dollar_volume["close"]
    stock_vol = stock_dollar_volume.groupby("ticker")["dollar_volume"].sum()

    # Calculate ETF dollar volume
    etf_dollar_volume = (window_df_etf["volume"] * window_df_etf["close"]).sum()

    return stock_vol / etf_dollar_volume


def compute_volatility_ratio(df_stock, df_etf, start_time, end_time):
    window_df_stock = df_stock[
        (df_stock["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_stock["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    window_df_etf = df_etf[
        (df_etf["window_start"].dt.time >= pd.to_datetime(start_time).time())
        & (df_etf["window_start"].dt.time <= pd.to_datetime(end_time).time())
    ]
    stock_vol = window_df_stock.groupby("ticker")["open"].std()
    etf_vol = window_df_etf["open"].std()  # Single ETF value
    return stock_vol / etf_vol


def compute_acceleration_diff(df_stock, df_etf):
    accel_stock = get_price_acceleration(df_stock)
    accel_etf = get_price_acceleration(df_etf)  # Returns tuple of scalars for ETF

    etf_values = [series.iloc[0] if not series.empty else np.nan for series in accel_etf]
    return tuple(stock_series - etf_val for stock_series, etf_val in zip(accel_stock, etf_values))


def compute_vwap_deviation_diff(df_stock, df_etf):
    vwap_stock = get_intraday_vwap_features(df_stock)["intraday_ratio_vwap_full_day_to_open"] - 1
    vwap_etf_dict = get_intraday_vwap_features(df_etf)
    vwap_etf_series = vwap_etf_dict["intraday_ratio_vwap_full_day_to_open"]
    vwap_etf = vwap_etf_series.iloc[0] - 1 if not vwap_etf_series.empty else np.nan
    return vwap_stock - vwap_etf


def compute_slope_diff(df_stock, df_etf):
    slope_stock = get_additional_momentum_features(df_stock)["intraday_intra_slope"]
    slope_etf_series = get_additional_momentum_features(df_etf)["intraday_intra_slope"]
    slope_etf = slope_etf_series.iloc[0] if not slope_etf_series.empty else np.nan
    return slope_stock - slope_etf


def get_etf_comparative_features(df_stock, df_etf, etf_ticker):
    data = {}
    # Return Relative to ETF
    data[f"intraday_return_relative_to_{etf_ticker}"] = compute_relative_return(df_stock, df_etf, "09:30", "15:55")
    data[f"intraday_last_hour_return_relative_to_{etf_ticker}"] = compute_relative_return(
        df_stock, df_etf, "15:00", "15:55"
    )
    data[f"intraday_last_ten_minutes_return_relative_to_{etf_ticker}"] = compute_relative_return(
        df_stock, df_etf, "15:45", "15:55"
    )
    data[f"intraday_last_five_minutes_return_relative_to_{etf_ticker}"] = compute_relative_return(
        df_stock, df_etf, "15:50", "15:55"
    )
    # Volume Ratio to ETF
    data[f"intraday_last_hour_volume_ratio_to_{etf_ticker}"] = compute_volume_ratio(df_stock, df_etf, "15:00", "15:55")
    # Volatility Ratio to ETF
    data[f"intraday_full_day_volatility_ratio_to_{etf_ticker}"] = compute_volatility_ratio(
        df_stock, df_etf, "09:30", "15:55"
    )
    # Price Acceleration Difference
    accel_diffs = compute_acceleration_diff(df_stock, df_etf)
    data[f"intraday_last5_acceleration_diff_to_{etf_ticker}"] = accel_diffs[0]
    data[f"intraday_last_hour_acceleration_diff_to_{etf_ticker}"] = accel_diffs[2]
    # VWAP Deviation Difference
    data[f"intraday_vwap_deviation_diff_to_{etf_ticker}"] = compute_vwap_deviation_diff(df_stock, df_etf)

    # Add close-to-VWAP difference
    momentum_stock = get_additional_momentum_features(df_stock)
    momentum_etf = get_additional_momentum_features(df_etf)
    close_to_vwap_diff = momentum_stock["intraday_close_to_vwap"] - momentum_etf["intraday_close_to_vwap"].iloc[0]
    data[f"intraday_close_to_vwap_diff_to_{etf_ticker}"] = close_to_vwap_diff

    # Momentum Slope Difference
    data[f"intraday_slope_diff_to_{etf_ticker}"] = compute_slope_diff(df_stock, df_etf)
    return data


# =============================================================================
# SECTION 5: ROLLING FEATURES (unchanged)
# =============================================================================


def add_rolling_ratio_features(
    df_features, feats_to_roll=None, return_feats=None, windows=[5, 10, 20], epsilon=1e-9, shift_by=1, n_workers=None
):
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
    if not isinstance(df_features.index, pd.MultiIndex):
        df_features = df_features.set_index(["ticker", "trade_date"])
    original_index = df_features.index
    original_columns = df_features.columns.tolist()
    groups = [group for _, group in df_features.groupby(level="ticker")]
    args = [
        (group, feats_to_roll, return_feats, windows, epsilon, shift_by, auction_volume_features) for group in groups
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_ticker_group, args))
    df_result = pd.concat(results, axis=0)
    columns_to_drop = []
    for w in windows:
        for feat in feats_to_roll:
            columns_to_drop.append(f"roll{w}_mean_{feat}")
        for feat in return_feats:
            columns_to_drop.extend([f"roll{w}_mean_{feat}", f"roll{w}_std_{feat}"])
    df_result.drop(columns=columns_to_drop, inplace=True)
    df_final = pd.concat([df_features[original_columns], df_result], axis=1)
    df_final = df_final.sort_index()
    if not isinstance(original_index, pd.MultiIndex):
        df_final = df_final.reset_index()
    return df_final


def process_ticker_group(args):
    group, feats_to_roll, return_feats, windows, epsilon, shift_by, auction_volume_features = args
    results = {}
    for feat in auction_volume_features:
        rolling_mean_name = f"roll5_mean_{feat}"
        rolling_mean = group[feat].fillna(0).rolling(window=5, min_periods=5).mean()
        results[rolling_mean_name] = rolling_mean.shift(shift_by)
    for w in windows:
        for feat in feats_to_roll:
            ratio_col_name = f"roll{w}_ratio_{feat}"
            rolling_mean_name = f"roll{w}_mean_{feat}"
            rolling_mean = group[feat].rolling(window=w, min_periods=w).mean()
            shifted_mean = rolling_mean.shift(shift_by)
            results[rolling_mean_name] = shifted_mean
            results[ratio_col_name] = group[feat] / (shifted_mean + epsilon)
    for w in windows:
        for feat in return_feats:
            mean_col = f"roll{w}_mean_{feat}"
            std_col = f"roll{w}_std_{feat}"
            z_col = f"zscore_{w}_{feat}"
            rolling_mean = group[feat].rolling(window=w, min_periods=w).mean()
            rolling_std = group[feat].rolling(window=w, min_periods=w).std()
            shifted_mean = rolling_mean.shift(shift_by)
            shifted_std = rolling_std.shift(shift_by)
            results[mean_col] = shifted_mean
            results[std_col] = shifted_std
            results[z_col] = (group[feat] - shifted_mean) / shifted_std.replace(0, np.nan)
    result_df = pd.DataFrame(results, index=group.index)
    return result_df
