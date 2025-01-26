import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ------------------------------------------------------------------------------
# SECTION 2: HELPER UTILITIES
# ------------------------------------------------------------------------------


def calc_return_for_window(df_window):
    """
    Given a subset of the DataFrame for a time window,
    compute the return = (last_close / first_open - 1).
    Returns a Series of returns indexed by 'ticker'.
    """
    if df_window.empty:
        return pd.Series(dtype="float64")  # empty series if no data
    grouped_open = df_window.groupby("ticker")["open"].first()
    grouped_close = df_window.groupby("ticker")["close"].last()
    return (grouped_close / grouped_open - 1).fillna(0)


def get_intraday_last_close_price_before_1555(df):
    df_last_close_before_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()]
    return df_last_close_before_1555.groupby("ticker")["close"].last()


# ------------------------------------------------------------------------------
# SECTION 3.1: VOLUME FEATURES
# ------------------------------------------------------------------------------


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
    """
    Computes volume-based features, focusing on the last hour, half hour,
    quarter hour, and the first hour.
    Returns (Series, Series, Series, Series, Series).
    """

    # 1) Time slicing
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

    # 2) Dollar volume in each window
    last_hour_dollar_volume = (df_last_hour["volume"] * df_last_hour["close"]).groupby(df_last_hour["ticker"]).sum()
    last_half_hour_dollar_volume = (
        (df_last_half_hour["volume"] * df_last_half_hour["close"]).groupby(df_last_half_hour["ticker"]).sum()
    )
    last_quarter_hour_dollar_volume = (
        (df_last_quarter_hour["volume"] * df_last_quarter_hour["close"]).groupby(df_last_quarter_hour["ticker"]).sum()
    )
    first_hour_dollar_volume = (df_first_hour["volume"] * df_first_hour["close"]).groupby(df_first_hour["ticker"]).sum()

    # 3) Total dollar volume until 15:55
    df_until_1555 = df[df["window_start"].dt.time <= pd.to_datetime("15:55").time()]
    total_dollar_volume_until_1555 = (
        (df_until_1555["volume"] * df_until_1555["close"]).groupby(df_until_1555["ticker"]).sum()
    )

    # Full day including pre/post market
    total_dollar_volume_all = (df["volume"] * df["close"]).groupby(df["ticker"]).sum()

    # 4) Ratios
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


# ------------------------------------------------------------------------------
# SECTION 3.2: VOLATILITY FEATURES
# ------------------------------------------------------------------------------


def get_intraday_volatility_features(df):
    """
    Computes the std. deviation (volatility) of 'open' for various windows:
    - full day, last hour, half hour, quarter hour.
    Returns four Series indexed by 'ticker'.
    """

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


# ------------------------------------------------------------------------------
# SECTION 3.3: PRICE FEATURES
# ------------------------------------------------------------------------------


def get_intraday_price_features(df):
    """
    Returns multiple price-based features, including:
      - daily_range
      - mean_intraday_range, median_intraday_range
      - intraday_return (09:30→15:55)
      - last_hour_return (15:00→15:55)
      - last_half_hour_return (15:30→15:55)
      - last_quarter_hour_return (15:45→15:55)
      - last_five_minutes_return (15:50→15:55)
      - first_hour_return (09:30→10:30)
      - first_half_hour_return (09:30→10:00)
      - first_five_minutes_return (09:30→09:35)
      - first_half_day_return (09:30→13:00)
      - second_half_day_return (13:00→15:55)
      - mid_day_return (11:30→14:00)
    """

    # A) Daily Range, Intraday Return
    df_intraday = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]

    # 1) Compute daily high & low
    grouped_high = df_intraday.groupby("ticker")["high"].max()
    grouped_low = df_intraday.groupby("ticker")["low"].min()
    daily_range = (grouped_high - grouped_low).fillna(0) / grouped_low.replace(0, pd.NA)

    # 2) Compute intraday range as (max - min)/first_low
    first_low = df_intraday.groupby("ticker")["low"].first()
    intraday_range = ((grouped_high - grouped_low) / first_low).fillna(0)
    mean_intraday_range = intraday_range.mean()
    median_intraday_range = intraday_range.median()

    # 3) Intraday return = close(15:55) / open(09:30) - 1
    df_930 = df[df["window_start"].dt.time == pd.to_datetime("09:30").time()].groupby("ticker")
    df_1555 = df[df["window_start"].dt.time == pd.to_datetime("15:55").time()].groupby("ticker")

    open_930 = df_930["open"].first()
    close_1555 = df_1555["close"].last()
    intraday_return = ((close_1555 / open_930) - 1).fillna(0)

    # B) Partial-window returns using our helper
    # Last hour (15:00→15:55)
    df_last_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_hour_return = calc_return_for_window(df_last_hour)

    # Last half hour (15:30→15:55)
    df_last_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_half_hour_return = calc_return_for_window(df_last_half_hour)

    # Last quarter hour (15:45→15:55)
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

    # Last five minutes (15:50→15:55)
    df_last_five_minutes = df[
        (df["window_start"].dt.time >= pd.to_datetime("15:50").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    last_five_minutes_return = calc_return_for_window(df_last_five_minutes)

    # First hour (09:30→10:30)
    df_first_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("10:30").time())
    ]
    first_hour_return = calc_return_for_window(df_first_hour)

    # First half hour (09:30→10:00)
    df_first_half_hour = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("10:00").time())
    ]
    first_half_hour_return = calc_return_for_window(df_first_half_hour)

    # First five minutes (09:30→09:35)
    df_first_five_minutes = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("09:35").time())
    ]
    first_five_minutes_return = calc_return_for_window(df_first_five_minutes)

    # First half day (09:30→13:00)
    df_first_half_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("13:00").time())
    ]
    first_half_day_return = calc_return_for_window(df_first_half_day)

    # Second half day (13:00→15:55)
    df_second_half_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("13:00").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ]
    second_half_day_return = calc_return_for_window(df_second_half_day)

    # Mid day (11:30→14:00)
    df_mid_day = df[
        (df["window_start"].dt.time >= pd.to_datetime("11:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("14:00").time())
    ]
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
    )


# ------------------------------------------------------------------------------
# SECTION 3.4: TRANSACTION FEATURES
# ------------------------------------------------------------------------------


def get_intraday_transaction_features(df):
    """
    Leverages the `transactions` column to create:
      - transaction count ratios for last hour, half hour, quarter hour
      - average transaction size (volume / transactions) for various windows
      - transaction volatility (std dev of 'transactions')
    """
    # A) Add per-bar average transaction size
    df["avg_txn_size"] = df["volume"] / df["transactions"].replace(0, pd.NA)

    # B) Time slicing
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

    # C) Transaction Count Ratios
    total_txn_until_1555 = df_until_1555.groupby("ticker")["transactions"].sum()
    last_hour_transactions = df_last_hour.groupby("ticker")["transactions"].sum()
    last_half_hour_transactions = df_last_half_hour.groupby("ticker")["transactions"].sum()
    last_quarter_hour_transactions = df_last_quarter_hour.groupby("ticker")["transactions"].sum()

    last_hour_transactions_ratio = (last_hour_transactions / total_txn_until_1555).fillna(0)
    last_half_hour_transactions_ratio = (last_half_hour_transactions / total_txn_until_1555).fillna(0)
    last_quarter_hour_transactions_ratio = (last_quarter_hour_transactions / total_txn_until_1555).fillna(0)

    # D) Average Transaction Size for each window
    avg_txn_size_full_day = df_full_day.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_hour = df_last_hour.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_half_hour = df_last_half_hour.groupby("ticker")["avg_txn_size"].mean()
    avg_txn_size_last_quarter_hour = df_last_quarter_hour.groupby("ticker")["avg_txn_size"].mean()

    # E) Transaction Volatility (std dev)
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


# ------------------------------------------------------------------------------
# SECTION 3.5: VWAP FEATURES
# ------------------------------------------------------------------------------


def get_intraday_vwap_features(df):
    """
    Compute standard VWAP from 09:30→15:55 plus partial-window VWAP.
    Also compute total dollar volume in each window.
    """
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

    # ratio_vwap_to_open = open(09:30) / vwap_full_day
    # (You can invert if you prefer.)
    open_930 = df_full_day.groupby("ticker")["open"].first()
    ratio_vwap_to_open = open_930 / vwap_full_day

    # Last-hour VWAP
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


# ------------------------------------------------------------------------------
# SECTION 3.6: ADDITIONAL MOMENTUM FEATURES
# ------------------------------------------------------------------------------


def get_additional_momentum_features(df):
    """
    More specialized momentum/trend features that combine price & time.
    Examples:
      - slope from open to close (approx)
      - final close vs. VWAP
      - last-hour bar counts
    """
    df_intraday = df[
        (df["window_start"].dt.time >= pd.to_datetime("09:30").time())
        & (df["window_start"].dt.time <= pd.to_datetime("15:55").time())
    ].copy()
    df_intraday.sort_values(["ticker", "window_start"], inplace=True)

    # Bar counts
    df_last_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:00").time()]
    df_last_half_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:30").time()]
    df_last_quarter_hour = df_intraday[df_intraday["window_start"].dt.time >= pd.to_datetime("15:45").time()]

    last_hour_bar_count = df_last_hour.groupby("ticker").size()
    last_half_hour_bar_count = df_last_half_hour.groupby("ticker").size()
    last_quarter_hour_bar_count = df_last_quarter_hour.groupby("ticker").size()

    # Slope from open(09:30) to close(15:55)
    def slope_calc(group):
        if group.empty:
            return pd.Series({"intra_slope": 0, "bar_count": 0})
        open_price = group.iloc[0]["open"]
        close_price = group.iloc[-1]["close"]
        bar_count = len(group)
        slope_val = (close_price - open_price) / bar_count if bar_count > 0 else 0
        return pd.Series({"intra_slope": slope_val, "bar_count": bar_count})

    slope_df = df_intraday.groupby("ticker").apply(slope_calc)
    slope_df = slope_df[["intra_slope", "bar_count"]]

    # Price to VWAP
    df_intraday["typical_price"] = (df_intraday["high"] + df_intraday["low"] + df_intraday["close"]) / 3.0
    df_intraday["dollar_volume"] = df_intraday["typical_price"] * df_intraday["volume"]
    grp = df_intraday.groupby("ticker")
    sum_dv = grp["dollar_volume"].sum()
    sum_vol = grp["volume"].sum()
    vwap = sum_dv / sum_vol

    final_close = df_intraday.groupby("ticker")["close"].last()
    price_to_vwap = ((final_close - vwap) / vwap).fillna(0)

    return {
        "intraday_intra_slope": slope_df["intra_slope"],
        "intraday_bar_count": slope_df["bar_count"],
        "intraday_close_to_vwap": price_to_vwap,
        "intraday_last_hour_bar_count": last_hour_bar_count,
        "intraday_last_half_hour_bar_count": last_half_hour_bar_count,
        "intraday_last_quarter_hour_bar_count": last_quarter_hour_bar_count,
    }


# ------------------------------------------------------------------------------
# SECTION 3.7: EXAMPLE OF A HELPER FOR FIRST/LAST BAR RATIOS
# ------------------------------------------------------------------------------


def ratio_first_open_to_last_close_bar_before_close(df):
    """
    Example: ratio of the last close before 15:50 to the first open after 09:30.
    """
    df_before_1550 = df[df["window_start"].dt.time < pd.to_datetime("15:50").time()]
    df_at_0930 = df[df["window_start"].dt.time >= pd.to_datetime("09:30").time()]

    last_bar_before_1550 = df_before_1550.groupby("ticker")["close"].last()
    first_bar_after_0930 = df_at_0930.groupby("ticker")["open"].first()

    ratio_first_to_last_bar_before_close = last_bar_before_1550 / first_bar_after_0930
    return ratio_first_to_last_bar_before_close


# ------------------------------------------------------------------------------
# SECTION 4: FINAL ASSEMBLER
# ------------------------------------------------------------------------------


def assemble_all_intraday_features(df):
    """
    For a given single-day DataFrame, compute:
      - volume features
      - volatility features
      - price-based intraday features
      - transaction-based features
      - VWAP features
      - momentum features
    and combine them into one DataFrame (one row per ticker).
    """

    last_close_before_1555 = get_intraday_last_close_price_before_1555(df)

    open_auction_dollar_volume, close_auction_dollar_volume = get_open_close_auction_dollar_volume(df)

    # A) Volume
    (
        last_hour_dollar_volume_ratio,
        last_half_hour_dollar_volume_ratio,
        last_quarter_hour_dollar_volume_ratio,
        first_hour_dollar_volume_ratio,
        total_dollar_volume_until_1555,
        total_dollar_volume_all,
    ) = get_intraday_volume_features(df)

    # B) Volatility
    (last_hour_volatility, last_half_hour_volatility, last_quarter_hour_volatility, full_day_volatility) = (
        get_intraday_volatility_features(df)
    )

    # C) Price
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
    ) = get_intraday_price_features(df)

    # D) Transactions
    txn_dict = get_intraday_transaction_features(df)

    # E) VWAP
    vwap_dict = get_intraday_vwap_features(df)

    # F) Momentum
    momentum_dict = get_additional_momentum_features(df)

    # G) Example ratio
    ratio_first_to_last_bar_at_1550 = ratio_first_open_to_last_close_bar_before_close(df)

    # Combine them into a dictionary of Series
    data = {
        "intraday_open_auction_dollar_volume": open_auction_dollar_volume,
        "intraday_close_auction_dollar_volume": close_auction_dollar_volume,
        "intraday_last_hour_dollar_volume_ratio": last_hour_dollar_volume_ratio,
        "intraday_last_half_hour_dollar_volume_ratio": last_half_hour_dollar_volume_ratio,
        "intraday_last_quarter_hour_dollar_volume_ratio": last_quarter_hour_dollar_volume_ratio,
        "intraday_first_hour_dollar_volume_ratio": first_hour_dollar_volume_ratio,
        "intraday_total_dollar_volume_until_1555": total_dollar_volume_until_1555,
        "intraday_total_dollar_volume_all": total_dollar_volume_all,
        "intraday_last_hour_volatility": last_hour_volatility,
        "intraday_last_half_hour_volatility": last_half_hour_volatility,
        "intraday_last_quarter_hour_volatility": last_quarter_hour_volatility,
        "intraday_full_day_volatility": full_day_volatility,
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
        "intraday_ratio_first_open_to_last_close_bar": ratio_first_to_last_bar_at_1550,
        "intraday_last_close_before_1555": last_close_before_1555,
    }

    # Merge in transaction-based features
    data.update(txn_dict)

    # Merge in VWAP features
    data.update(vwap_dict)

    # Merge in momentum features
    data.update(momentum_dict)

    combined_df = pd.DataFrame(data)
    combined_df.index.name = "ticker"
    return combined_df


def add_rolling_ratio_features(
    df_features, feats_to_roll=None, return_feats=None, windows=[5, 10, 20], epsilon=1e-9, shift_by=1, n_workers=None
):
    """
    Optimized version using parallel processing and vectorized operations.
    """
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
    if not isinstance(df_features.index, pd.MultiIndex):
        df_features = df_features.set_index(["ticker", "trade_date"])

    # Store original index for later
    original_index = df_features.index

    # Store original columns
    original_columns = df_features.columns.tolist()

    # Prepare arguments for parallel processing
    groups = [group for _, group in df_features.groupby(level="ticker")]
    args = [
        (group, feats_to_roll, return_feats, windows, epsilon, shift_by, auction_volume_features) for group in groups
    ]

    # Process groups in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_ticker_group, args))

    # Combine results
    df_result = pd.concat(results, axis=0)

    # Calculate columns to drop (intermediate calculations)
    columns_to_drop = []
    for w in windows:
        for feat in feats_to_roll:
            columns_to_drop.append(f"roll{w}_mean_{feat}")
        for feat in return_feats:
            columns_to_drop.extend([f"roll{w}_mean_{feat}", f"roll{w}_std_{feat}"])

    # Drop intermediate columns
    df_result.drop(columns=columns_to_drop, inplace=True)

    # Merge with original features
    df_final = pd.concat([df_features[original_columns], df_result], axis=1)

    # Ensure the index is sorted
    df_final = df_final.sort_index()

    # Reset index if it wasn't originally a MultiIndex
    if not isinstance(original_index, pd.MultiIndex):
        df_final = df_final.reset_index()

    return df_final


def process_ticker_group(args):
    """
    Process a single ticker group in parallel.
    """
    group, feats_to_roll, return_feats, windows, epsilon, shift_by, auction_volume_features = args

    # Pre-allocate results dictionary
    results = {}

    # 1) Process auction volume features (only window=5)
    for feat in auction_volume_features:
        rolling_mean_name = f"roll5_mean_{feat}"
        rolling_mean = group[feat].fillna(0).rolling(window=5, min_periods=5).mean()
        results[rolling_mean_name] = rolling_mean.shift(shift_by)

    # 2) Create ratio features
    for w in windows:
        for feat in feats_to_roll:
            ratio_col_name = f"roll{w}_ratio_{feat}"
            rolling_mean_name = f"roll{w}_mean_{feat}"

            rolling_mean = group[feat].rolling(window=w, min_periods=w).mean()
            shifted_mean = rolling_mean.shift(shift_by)
            results[rolling_mean_name] = shifted_mean
            results[ratio_col_name] = group[feat] / (shifted_mean + epsilon)

    # 3) Create z-score features
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

    # Convert results to DataFrame
    result_df = pd.DataFrame(results, index=group.index)

    return result_df


"""
                                                           0                8841   ...            26489          35322
ticker                                                         A                A  ...                A              A
intraday_last_hour_dollar_volume_ratio                  0.260378         0.396521  ...         0.281574       0.296676
intraday_last_half_hour_dollar_volume_ratio             0.185829         0.310729  ...         0.149395       0.201539
intraday_last_quarter_hour_dollar_volume_ratio          0.109213         0.221568  ...         0.100951       0.129701
intraday_first_hour_dollar_volume_ratio                   0.2189         0.195277  ...         0.193971       0.131365
intraday_total_dollar_volume_until_1555           120800590.0652    87625819.4565  ...     96954636.914  84755785.4584
intraday_last_hour_volatility                           0.197447         0.183028  ...         0.125166       0.182242
intraday_last_half_hour_volatility                      0.228005         0.133564  ...         0.129827       0.124198
intraday_last_quarter_hour_volatility                   0.245915         0.120363  ...         0.142272       0.152764
intraday_full_day_volatility                            0.393774         0.357051  ...         0.422278       0.609969
intraday_daily_range                                    0.019944          0.01856  ...         0.020607       0.027419
intraday_mean_intraday_range                            0.037241         0.034862  ...          0.03959       0.037135
intraday_median_intraday_range                            0.0266         0.025508  ...         0.027455       0.025168
intraday_return                                         0.016905         0.007329  ...        -0.000181       0.018238
intraday_last_hour_return                               0.001999         0.002056  ...          0.00154      -0.000886
intraday_last_half_hour_return                          0.001332         0.002431  ...         0.000362      -0.001859
intraday_last_quarter_hour_return                       0.007609        -0.000652  ...         0.001812       -0.00371
intraday_last_five_minutes_return                        0.00324        -0.000559  ...         0.001903      -0.004238
intraday_first_hour_return                              0.012365         0.011651  ...         0.003527       0.012189
intraday_first_half_hour_return                         0.005265         0.005403  ...        -0.002713       0.003611
intraday_first_five_minutes_return                      0.001739        -0.000188  ...         0.002261      -0.000903
intraday_first_half_day_return                          0.011737         0.010805  ...         -0.00624       0.013904
intraday_second_half_day_return                         0.004006        -0.003439  ...         0.005914       0.004543
intraday_mid_day_return                                -0.003923        -0.006035  ...         -0.00136       0.006865
intraday_ratio_first_to_last_bar_at_1550:00             1.011689         1.010054  ...         0.998101       1.022842
intraday_last_hour_transactions_ratio                   0.244614         0.356376  ...          0.25702       0.295914
intraday_last_half_hour_transactions_ratio              0.161014         0.261864  ...          0.15832       0.193096
intraday_last_quarter_hour_transactions_ratio           0.090042         0.171925  ...         0.098578       0.115946
intraday_avg_txn_size_full_day                          55.81937        48.874455  ...        50.640261      53.308151
intraday_avg_txn_size_last_hour                        60.150606        53.134083  ...        56.498986       52.57118
intraday_avg_txn_size_last_half_hour                   67.529273        59.195566  ...        49.203742      55.408937
intraday_avg_txn_size_last_quarter_hour                73.188112        67.219348  ...        56.183508      60.950143
intraday_txn_volatility_full_day                       37.187043        43.799684  ...        29.595837      27.176766
intraday_avg_txn_size_volatility_full_day              30.344512        32.039299  ...         39.12111      31.688353
intraday_ratio_vwap_full_day_to_open                    0.989747          0.99268  ...         1.000605       0.984904
intraday_ratio_vwap_last_hour_to_open                   1.001173         0.998437  ...         0.999404       0.999058
intraday_dollar_volume_full_day                 120787605.068533    87615206.4476  ...  96940542.101767  84760491.3033
intraday_dollar_volume_last_hour                 31453384.061667  34747530.426033  ...  27297782.162433  25147267.6974
intraday_intra_slope                                    0.004581         0.002102  ...        -0.000054        0.00543
intraday_bar_count                                         382.0            371.0  ...            370.0          372.0
intraday_close_to_vwap                                  0.006478        -0.000045  ...         0.000424       0.002866
intraday_last_5min_return                                0.00324        -0.000559  ...         0.001903      -0.004238
intraday_last_10min_return                              0.007609        -0.000652  ...         0.001812       -0.00371
intraday_last_hour_bar_count                                56.0             56.0  ...             56.0           56.0
intraday_last_half_hour_bar_count                           26.0             26.0  ...             26.0           26.0
intraday_last_quarter_hour_bar_count                        11.0             11.0  ...             11.0           11.0
trade_date                                            2020-11-02       2020-11-03  ...       2020-11-05     2020-11-06

[47 rows x 5 columns]

"""
