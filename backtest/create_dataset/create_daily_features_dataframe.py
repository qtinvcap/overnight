from features.daily.features_engineering import compute_advanced_daily_features
from features.daily.retrieve_price_data_api import PolygonHistoricalDailyData
from features.utils import get_ticker_full_tickers_list
import os
import time

if __name__ == "__main__":

    api_key = os.getenv("POLYGON_API_KEY")
    start_time = time.time()

    print("Creating directory structure...")
    os.makedirs("data/daily", exist_ok=True)

    print("Retrieving ticker list...")
    full_tickers = get_ticker_full_tickers_list()
    print(f"Retrieved {len(full_tickers)} tickers in {time.time() - start_time:.2f} seconds")

    print("\nInitializing Polygon API connection...")
    polygon = PolygonHistoricalDailyData(api_key=api_key)

    print("Fetching historical data from Polygon API...")
    api_start_time = time.time()
    df_daily = polygon.get_historical_data("2003-01-01", "2025-01-16")
    print(f"Retrieved {len(df_daily)} rows of historical data in {time.time() - api_start_time:.2f} seconds")

    print("\nFiltering data for selected tickers...")
    filter_start_time = time.time()
    df_daily = df_daily[df_daily["ticker"].isin(full_tickers)]
    print(f"Filtered data to {len(df_daily)} rows in {time.time() - filter_start_time:.2f} seconds")

    print("\nComputing advanced features...")
    features_start_time = time.time()
    df_features = compute_advanced_daily_features(df_daily)
    print(f"Computed features in {time.time() - features_start_time:.2f} seconds")

    print("\nSaving features to parquet file...")
    save_start_time = time.time()
    df_features.to_parquet("/home/aime/overnigh_strat/overnight/data/daily/daily_features.parquet")
    print(f"Saved features in {time.time() - save_start_time:.2f} seconds")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
