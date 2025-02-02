import pandas as pd
import os
import time
from overnight.features.nbbo.retrieve_nbbo_data import main
from overnight.features.utils import get_ticker_full_tickers_list


if __name__ == "__main__":
    print("Starting...")
    tickers = list(
        pd.read_parquet("/home/aime/overnigh_strat/overnight/data/final_features.parquet")["ticker"].unique()
    )
    print(f"Found {len(tickers)} tickers")
    start_date_str = "2024-06-01"
    end_date_str = "2024-06-10"
    print(f"Processing {start_date_str} to {end_date_str}")
    main(tickers, start_date_str, end_date_str)
    print("Done")
