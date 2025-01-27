import pandas as pd
import time
from features.utils import get_ticker_full_tickers_list
from features.combine_intraday_daily_features import (
    merge_daily_and_intraday_data,
    add_temporal_features,
    add_indicator_normalizations,
)

from features.vix import generate_vix_data_and_merge
from features.intraday.features_engineering import add_rolling_ratio_features
import os


if __name__ == "__main__":
    path_to_overnight_trading = os.getenv("OVERNIGHT_TRADING_PATH")
    os.makedirs("data", exist_ok=True)

    start_time = time.time()
    tickers = get_ticker_full_tickers_list()
    print(f"Getting tickers list took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_daily = pd.read_parquet(f"{path_to_overnight_trading}/data/daily/daily_features.parquet")
    df_intraday = pd.read_parquet(f"{path_to_overnight_trading}/data/intraday/intraday_features.parquet")
    print(f"Loading parquet files took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_daily = df_daily[df_daily["ticker"].isin(tickers)]
    df_intraday = df_intraday[df_intraday["ticker"].isin(tickers)]
    print(f"Filtering tickers took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_intraday = add_rolling_ratio_features(df_intraday)
    print(f"Adding rolling ratio features took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_daily = generate_vix_data_and_merge(df_daily)
    print(f"Generating VIX data took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_combined = merge_daily_and_intraday_data(df_daily, df_intraday)
    print(f"Merging daily and intraday data took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_combined = add_indicator_normalizations(df_combined)
    print(f"Adding indicator normalizations took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_combined = add_temporal_features(df_combined)
    print(f"Adding temporal features took: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    df_combined.to_parquet("data/final_features.parquet")
    print(f"Saving final parquet file took: {time.time() - start_time:.2f} seconds")
