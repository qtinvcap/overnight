import os
import gzip
import io
import time
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import boto3
from botocore.config import Config
import pandas_market_calendars as mcal
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from dask import delayed, compute
import dask

###############################################################################
# S3 CLIENT SETUP
###############################################################################
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set")

session = boto3.Session(
    aws_access_key_id="0b8566f5-0f1b-41b7-b947-af9dcdea0a87",
    aws_secret_access_key="hOb5i0V1zyzhylPRWN4KA29OJqpFObTC",
)
s3 = session.client(
    "s3",
    endpoint_url="https://files.polygon.io",
    config=Config(
        signature_version="s3v4", read_timeout=300, connect_timeout=30, retries={"max_attempts": 10, "mode": "standard"}
    ),
)
bucket_name = "flatfiles"
prefix = "us_stocks_sip"  # Adjust if needed


###############################################################################
# NBBO FEATURE COMPUTATION
###############################################################################
def compute_nbbo_features(quotes):
    if not quotes:
        return {
            "avg_spread": np.nan,
            "max_spread": np.nan,
            "spread_volatility": np.nan,
            "mid_price_volatility": np.nan,
            "avg_mid_price": np.nan,
            "median_mid_price": np.nan,
            "avg_dollar_liquidity_ask": np.nan,
            "median_dollar_liquidity_ask": np.nan,
            "min_dollar_liquidity_ask": np.nan,
            "max_dollar_liquidity_ask": np.nan,
            "avg_dollar_liquidity_bid": np.nan,
            "imbalance": np.nan,
            "quote_count": 0,
            "spread_skew": np.nan,
            "spread_kurtosis": np.nan,
            "weighted_spread": np.nan,
            "liquidity_ratio": np.nan,
        }
    ask_prices = []
    bid_prices = []
    dollar_at_asks = []
    dollar_at_bids = []
    spreads = []
    mid_prices = []
    for q in quotes:
        ask_price = q.get("ask_price", 0.0)
        bid_price = q.get("bid_price", 0.0)
        ask_size_rl = q.get("ask_size", 0)
        bid_size_rl = q.get("bid_size", 0)
        ask_size_sh = ask_size_rl * 100
        bid_size_sh = bid_size_rl * 100
        spread = ask_price - bid_price
        mid_price = (ask_price + bid_price) / 2.0 if (ask_price > 0 and bid_price > 0) else 0.0
        dollar_at_ask = ask_price * ask_size_sh
        dollar_at_bid = bid_price * bid_size_sh
        ask_prices.append(ask_price)
        bid_prices.append(bid_price)
        spreads.append(spread)
        mid_prices.append(mid_price)
        dollar_at_asks.append(dollar_at_ask)
        dollar_at_bids.append(dollar_at_bid)
    spreads_arr = np.array(spreads)
    mid_prices_arr = np.array(mid_prices)
    dollar_at_asks_arr = np.array(dollar_at_asks)
    dollar_at_bids_arr = np.array(dollar_at_bids)
    avg_spread = np.mean(spreads_arr)
    max_spread = np.max(spreads_arr)
    spread_volatility = np.std(spreads_arr)
    mid_price_volatility = np.std(mid_prices_arr)
    avg_mid_price = np.mean(mid_prices_arr)
    median_mid_price = np.median(mid_prices_arr)
    avg_dollar_ask = np.mean(dollar_at_asks_arr)
    median_dollar_ask = np.median(dollar_at_asks_arr)
    min_dollar_ask = np.min(dollar_at_asks_arr)
    max_dollar_ask = np.max(dollar_at_asks_arr)
    avg_dollar_bid = np.mean(dollar_at_bids_arr)
    sum_dollar_ask = np.sum(dollar_at_asks_arr)
    sum_dollar_bid = np.sum(dollar_at_bids_arr)
    if (sum_dollar_ask + sum_dollar_bid) != 0:
        imbalance = (sum_dollar_bid - sum_dollar_ask) / (sum_dollar_bid + sum_dollar_ask)
    else:
        imbalance = np.nan
    quote_count = len(quotes)
    std_spread = np.std(spreads_arr)
    if std_spread < 1e-8:
        spread_skew = 0.0
        spread_kurtosis = 0.0
    else:
        spread_skew = skew(spreads_arr)
        spread_kurtosis = kurtosis(spreads_arr)
    total_liquidity = dollar_at_asks_arr + dollar_at_bids_arr
    if total_liquidity.sum() > 0:
        weighted_spread = np.average(spreads_arr, weights=total_liquidity)
    else:
        weighted_spread = np.nan
    liquidity_ratio = sum_dollar_bid / sum_dollar_ask if sum_dollar_ask != 0 else np.nan
    features = {
        "avg_spread": avg_spread,
        "max_spread": max_spread,
        "spread_volatility": spread_volatility,
        "mid_price_volatility": mid_price_volatility,
        "avg_mid_price": avg_mid_price,
        "median_mid_price": median_mid_price,
        "avg_dollar_liquidity_ask": avg_dollar_ask,
        "median_dollar_liquidity_ask": median_dollar_ask,
        "min_dollar_liquidity_ask": min_dollar_ask,
        "max_dollar_liquidity_ask": max_dollar_ask,
        "avg_dollar_liquidity_bid": avg_dollar_bid,
        "imbalance": imbalance,
        "quote_count": quote_count,
        "spread_skew": spread_skew,
        "spread_kurtosis": spread_kurtosis,
        "weighted_spread": weighted_spread,
        "liquidity_ratio": liquidity_ratio,
    }
    return features


###############################################################################
# PROCESS FLAT FILE FOR ONE DAY (ALL TICKERS)
###############################################################################
def gather_nbbo_features_from_file(df, windows_dict, time_col="timestamp"):
    # Ensure the time column is datetime and localized to Eastern Time.
    if "timestamp" not in df.columns and "sip_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        df[time_col] = pd.to_datetime(df[time_col])
        if df[time_col].dt.tz is None:
            eastern = pytz.timezone("US/Eastern")
            df[time_col] = df[time_col].dt.tz_localize(eastern)
    window_dfs = {}
    for window_label, (start_time_str, end_time_str) in windows_dict.items():
        start_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
        end_time = datetime.strptime(end_time_str, "%H:%M:%S").time()
        df_window = df[df[time_col].dt.time.between(start_time, end_time)]
        groups = df_window.groupby("ticker")
        features_list = []
        for ticker, group in groups:
            quotes = group.to_dict(orient="records")
            feats = compute_nbbo_features(quotes)
            feats["ticker"] = ticker
            features_list.append(feats)
        window_df = pd.DataFrame(features_list).set_index("ticker")
        window_df = window_df.add_prefix(f"{window_label}_")
        window_dfs[window_label] = window_df
    final_df = None
    for wdf in window_dfs.values():
        if final_df is None:
            final_df = wdf
        else:
            final_df = final_df.join(wdf, how="outer")
    unique_dates = df[time_col].dt.date.unique()
    if len(unique_dates) > 0:
        final_df["date"] = unique_dates[0]
    return final_df


###############################################################################
# S3 DATA RETRIEVAL: Read One Day's File from S3 (with retries)
###############################################################################
def read_and_label_one_day(date_obj, retries=3, delay=5):
    date_str = date_obj.strftime("%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    object_key = f"{prefix}/quotes_v1/{year}/{month}/{date_str}.csv.gz"
    print(f"Fetching {object_key} ...")
    attempt = 0
    while attempt < retries:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            gz_body = response["Body"].read()
            # Specify tab delimiter if the file is tab-separated
            with gzip.GzipFile(fileobj=io.BytesIO(gz_body)) as gz_file:
                df = pd.read_csv(gz_file, low_memory=False)
            break
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed for {object_key}: {e}")
            time.sleep(delay)
    else:
        print(f"Failed to read file {object_key} after {retries} attempts")
        return None
    if df.empty:
        return None
    if "timestamp" not in df.columns and "sip_timestamp" not in df.columns:
        print(f"Error: Expected a 'timestamp' or 'sip_timestamp' column in the data for {object_key}.")
        return None
    return df


###############################################################################
# WORKER FUNCTION: Process One Day (by date string)
###############################################################################
def process_day(date_str, windows_dict):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        df_day = read_and_label_one_day(date_obj)
        if df_day is None:
            print(f"No data for {date_str}")
            return None
        df_features = gather_nbbo_features_from_file(df_day, windows_dict, time_col="timestamp")
        df_features["date"] = date_str
        return df_features
    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        return None


###############################################################################
# MAIN: Parallelize Across Multiple Days Using Dask Delayed
###############################################################################
from dask import delayed, compute


def main_dask():
    start_date_str = "2024-05-06"
    end_date_str = "2024-06-06"
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)
    all_dates = pd.to_datetime(schedule.index).strftime("%Y-%m-%d").tolist()

    windows_dict = {
        "morn": ("09:30:00", "10:00:00"),
        "mid": ("11:30:00", "13:00:00"),
        "lh": ("14:56:00", "15:56:00"),
        "auction": ("15:56:00", "16:00:00"),
    }

    delayed_results = [delayed(process_day)(date_str, windows_dict) for date_str in all_dates]
    results = compute(*delayed_results)
    # Filter out None results.
    results = [r for r in results if r is not None]
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.reset_index(drop=True)
        final_df.to_csv("nbbo_features_multiple_days_dask.csv", index=False)
        print("Saved nbbo_features_multiple_days_dask.csv")
        return final_df
    else:
        print("No data processed.")
        return None


if __name__ == "__main__":
    df = main_dask()
    if df is not None:
        print(df.head(10))
