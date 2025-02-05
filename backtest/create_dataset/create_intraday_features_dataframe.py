import pandas as pd
import os
import time
from pathlib import Path
from overnight.features.intraday.main_intraday import process_intraday_data
from overnight.features.intraday.features_engineering import add_rolling_ratio_features


if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting script at {time.strftime('%H:%M:%S')}")

    try:
        root_path = Path(os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")))
        print("Processing intraday data...")
        df_features = process_intraday_data("2004-01-02", "2025-01-31")

        data_dir = root_path / "data" / "intraday"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving dataframe with shape {df_features.shape}")
        df_features.to_parquet(data_dir / "intraday_features.parquet")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Script completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
