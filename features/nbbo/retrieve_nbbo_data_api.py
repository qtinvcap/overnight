import os
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Set up API key and base URL.
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set")
BASE_URL = "https://api.polygon.io/v3/quotes"


###############################################################################
# NBBO FEATURE COMPUTATION FUNCTION
###############################################################################
def compute_nbbo_features(quotes):
    """
    Given a list of NBBO quote dictionaries from Polygon, compute aggregate metrics.
    Returns a dictionary with keys such as:
      - avg_spread, max_spread, spread_volatility,
      - mid_price_volatility, avg_mid_price, median_mid_price,
      - avg_dollar_liquidity_ask, median_dollar_liquidity_ask, min_dollar_liquidity_ask, max_dollar_liquidity_ask,
      - avg_dollar_liquidity_bid,
      - imbalance, quote_count, spread_skew, spread_kurtosis, weighted_spread, liquidity_ratio.
    """
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
        # Assume 1 round lot = 100 shares.
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
# API QUOTES FETCHING & NBBO FEATURE PROCESSING FOR A SINGLE TICKER & DAY
###############################################################################
def process_ticker(ticker, date_str):
    """
    For a given ticker and date string, fetch NBBO quotes via the Polygon API,
    handle pagination (ensuring the API key is always included),
    compute NBBO features, and return a DataFrame row.
    """
    params = {
        "timestamp": date_str,  # e.g., "2024-06-06"
        "limit": 1000,
        "apiKey": API_KEY,
    }
    url = f"{BASE_URL}/{ticker}"
    all_quotes = []

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching {ticker} on {date_str}: {response.status_code} {response.text}")
            break
        data = response.json()
        if "results" in data:
            all_quotes.extend(data["results"])
        else:
            break

        if "next_url" in data and data["next_url"]:
            next_url = data["next_url"]
            # Ensure the API key is in the next_url.
            if "apiKey=" not in next_url:
                separator = "&" if "?" in next_url else "?"
                next_url = next_url + separator + "apiKey=" + API_KEY
            url = next_url
            params = {}  # Reset parameters since next_url should contain them now.
        else:
            break

    feats = compute_nbbo_features(all_quotes)
    feats["ticker"] = ticker
    feats["date"] = date_str
    return pd.DataFrame([feats])


###############################################################################
# MAIN: Process Multiple Tickers in Parallel Using Threads
###############################################################################
def main():
    # Specify the date for which you want to process data.
    date_str = "2024-06-06"
    # List of tickers to process; you can pass your full list here.
    tickers = ["AAPL", "TSLA", "GOOG"]

    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_ticker, ticker, date_str): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tickers"):
            df_ticker = future.result()
            if df_ticker is not None:
                results.append(df_ticker)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv("nbbo_features_api.csv", index=False)
        print("Saved nbbo_features_api.csv")
        return final_df
    else:
        print("No data processed.")
        return None


if __name__ == "__main__":
    df = main()
    if df is not None:
        print(df.head(10))
