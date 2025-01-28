import boto3
import pandas as pd
from botocore.config import Config
import pandas_market_calendars as mcal

from overnight.features.intraday.retrieve_intraday_data_flat_file import fetch_one_day
from overnight.features.intraday.features_engineering import assemble_all_intraday_features
from multiprocessing import Pool, cpu_count


def process_day_chunk(chunk_info):
    """
    Process a chunk of trading days and return the results
    Args:
        chunk_info: Tuple of (days_chunk, next_first_day) where
            days_chunk is a list of dates to process
            next_first_day is the first day of the next chunk (or None if last chunk)
    """
    days_chunk, chunk_id = chunk_info  # Modified to include chunk_id
    results = []

    print(f"Starting chunk {chunk_id}, processing {len(days_chunk)} days")

    for i, current_date in enumerate(days_chunk):
        if i % 120 == 0:  # Print progress every 5 days
            print(f"Chunk {chunk_id}: Processing day {i+1}/{len(days_chunk)} - {current_date}")

        df_dayN = fetch_one_day(current_date)
        if df_dayN is None or df_dayN.empty:
            continue

        # Process intraday features
        df_dayN_features = assemble_all_intraday_features(df_dayN)
        df_dayN_features["trade_date"] = current_date

        if df_dayN_features.index.name == "ticker":
            df_dayN_features = df_dayN_features.reset_index()

        if not df_dayN_features.empty:
            results.append(df_dayN_features)

    print(f"Finished chunk {chunk_id}, processed {len(results)} days with data")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def process_intraday_data(start_date_str, end_date_str):
    """
    Parallelized version of the data loading function
    """
    print(f"Starting process_intraday_data from {start_date_str} to {end_date_str}")

    # Get trading days
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)
    trading_days = mcal.date_range(schedule, frequency="1D")
    trading_days = [d.date() for d in trading_days]

    print(f"Found {len(trading_days)} trading days")

    # Split trading days into chunks for parallel processing
    n_cores = cpu_count() - 1  # Leave one core free
    chunk_size = max(1, len(trading_days) // n_cores)
    day_chunks = [trading_days[i : i + chunk_size] for i in range(0, len(trading_days), chunk_size)]

    print(f"Split into {len(day_chunks)} chunks using {n_cores} cores")

    # Create chunks with chunk IDs
    chunk_with_next = [(chunk, i) for i, chunk in enumerate(day_chunks)]

    # Process chunks in parallel
    print("Starting parallel processing...")
    with Pool(processes=n_cores) as pool:
        results = list(pool.imap(process_day_chunk, chunk_with_next))

    print("Combining results...")
    final_df = pd.concat([df for df in results if not df.empty], ignore_index=True)
    print(f"Final dataframe shape: {final_df.shape}")

    return final_df


if __name__ == "__main__":
    process_intraday_data("2025-01-27", "2025-01-27")
