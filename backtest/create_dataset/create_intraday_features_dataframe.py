import pandas as pd
import os
import time
from features.intraday.main_intraday import process_intraday_data
from features.intraday.features_engineering import add_rolling_ratio_features


if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting script at {time.strftime('%H:%M:%S')}")

    try:
        print("Processing intraday data...")
        df_features = process_intraday_data("2004-01-02", "2024-12-31")

        print(f"Saving dataframe with shape {df_features.shape}")
        df_features.to_parquet("data/intraday/intraday_features.parquet")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Script completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
