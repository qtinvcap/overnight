import pandas as pd


def merge_daily_and_intraday_data(daily_features, intraday_features):
    # Reset index to get ticker and trade_date as columns
    if isinstance(daily_features.index, pd.MultiIndex):
        daily_features = daily_features.reset_index()
    if isinstance(intraday_features.index, pd.MultiIndex):
        intraday_features = intraday_features.reset_index()

    daily_features.rename(columns={"date": "trade_date"}, inplace=True)
    intraday_features["trade_date"] = pd.to_datetime(intraday_features["trade_date"])
    daily_features["trade_date"] = pd.to_datetime(daily_features["trade_date"])

    final_merged = pd.merge(intraday_features, daily_features, on=["ticker", "trade_date"], how="left")
    return final_merged


def add_temporal_features(df):
    df["day_of_week"] = df["trade_date"].dt.dayofweek
    df["month"] = df["trade_date"].dt.month
    df["year"] = df["trade_date"].dt.year
    return df


def add_indicator_normalizations(df):
    """
    AT THE DATASET LEVEL
    """

    # ATR ratio
    df["atr_ratio"] = (
        df["atr"] / df["current_day_open"]
    )  # Using open as the price from intraday might be affected by stock split or dividends !!!!!!
    df.drop(columns=["atr"], inplace=True)

    # MACD ratio
    df["macd_line_ratio"] = df["macd_line"] / df["current_day_open"]
    df["macd_signal_ratio"] = df["macd_signal"] / df["current_day_open"]
    df["macd_hist_ratio"] = df["macd_hist"] / df["current_day_open"]
    df.drop(columns=["macd_line", "macd_signal", "macd_hist"], inplace=True)

    # Pivot ratios
    df["pivot_ratio"] = df["pivot"] / df["current_day_open"]
    df["r1_ratio"] = df["r1"] / df["current_day_open"]
    df["s1_ratio"] = df["s1"] / df["current_day_open"]
    df["r2_ratio"] = df["r2"] / df["current_day_open"]
    df["s2_ratio"] = df["s2"] / df["current_day_open"]

    # Drop price pivot support/resistance
    df.drop(columns=["pivot", "r1", "s1", "r2", "s2"], inplace=True)

    # OBV ratio
    df["obv_ratio"] = df["obv"] / (df["roll20_avg_dollar_volume"] + 1e-9)
    df.drop(columns=["obv"], inplace=True)
    # VOLUME RATIOS
    df["dollar_volume_intraday_vs_20d"] = df["intraday_total_dollar_volume_until_1555"] / df["roll20_avg_dollar_volume"]
    df["dollar_volume_intraday_vs_10d"] = df["intraday_total_dollar_volume_until_1555"] / df["roll10_avg_dollar_volume"]
    df["dollar_volume_intraday_vs_5d"] = df["intraday_total_dollar_volume_until_1555"] / df["roll5_avg_dollar_volume"]

    df["last_hour_volume_ratio_vs_20d"] = df["intraday_last_hour_dollar_volume_ratio"] / df["roll20_avg_dollar_volume"]

    # RANGE RATIOS
    df["intraday_range_vs_20d"] = df["intraday_daily_range"] / df["roll20_avg_daily_range"]
    df["intraday_range_vs_10d"] = df["intraday_daily_range"] / df["roll10_avg_daily_range"]
    df["intraday_range_vs_5d"] = df["intraday_daily_range"] / df["roll5_avg_daily_range"]

    # MORNING GAP + INTRADAY RETURN
    df["morning_gap_plus_intra"] = df["today_open_gap"] + df["intraday_return"]

    # RATIO MA
    df["is_short_above_long"] = (df["short_ma"] > df["long_ma"]).astype(int)
    df["is_short_above_extra_long"] = (df["short_ma"] > df["extra_long_ma"]).astype(int)

    df["ratio_ma_close_vs_short"] = df["current_day_open"] / df["short_ma"]
    df["ratio_ma_close_vs_long"] = df["current_day_open"] / df["long_ma"]
    df["ratio_ma_close_vs_extra_long"] = df["current_day_open"] / df["extra_long_ma"]

    return df
