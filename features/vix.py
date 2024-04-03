import pandas as pd
import numpy as np
import yfinance as yf
import os
from pathlib import Path


def load_vix_data():

    root_path = Path(os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")))
    vix = pd.read_csv(root_path / "price_data" / "VIX_History.csv")

    vix["DATE"] = pd.to_datetime(vix["DATE"]).dt.strftime("%Y-%m-%d")
    vix.rename(columns={"DATE": "date"}, inplace=True)

    return vix


def get_vix_data():
    vix = yf.Ticker("^VIX").history(period="max")
    vix.reset_index(inplace=True)
    vix.rename(columns={"Date": "date"}, inplace=True)
    vix.rename(columns={"Close": "CLOSE"}, inplace=True)
    vix.rename(columns={"Open": "OPEN"}, inplace=True)
    vix.rename(columns={"High": "HIGH"}, inplace=True)
    vix.rename(columns={"Low": "LOW"}, inplace=True)
    vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")
    return vix


def create_vix_features(vix_df, rolling_windows=[5, 10]):
    """Compute VIX-derived features: previous day levels, overnight return, and rolling averages."""
    df = vix_df.copy().sort_values("date").reset_index(drop=True)

    # Keep the original columns for clarity
    df["vix_open_current_day"] = df["OPEN"]

    # SHIFT the close, high, low by 1 day forward => so row T references T-1
    df["vix_close_prev_day"] = df["CLOSE"].shift(1)
    df["vix_high_prev_day"] = df["HIGH"].shift(1)
    df["vix_low_prev_day"] = df["LOW"].shift(1)

    # vix_range_prev_day => day T is about T-1 data
    df["vix_range_prev_day"] = (df["vix_high_prev_day"] - df["vix_low_prev_day"]) / df["vix_low_prev_day"].replace(
        0, np.nan
    )

    # vix_overnight_return => (vix_open_T / vix_close_{T-1}) - 1
    df["vix_overnight_return"] = (df["vix_open_current_day"] / df["vix_close_prev_day"].replace(0, np.nan)) - 1.0

    for w in rolling_windows:
        df[f"vix_roll{w}_close"] = df["vix_close_prev_day"].shift(1).rolling(window=w).mean()

    return df[
        ["date", "vix_open_current_day", "vix_close_prev_day", "vix_range_prev_day", "vix_overnight_return"]
        + [f"vix_roll{w}_close" for w in rolling_windows]
    ]


def get_vix_features():
    vix = get_vix_data()
    vix_features = create_vix_features(vix)
    return vix_features


def generate_vix_data_and_merge(daily_df):
    vix_df = get_vix_features()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    return pd.merge(daily_df, vix_df, on="date", how="left")


if __name__ == "__main__":

    root_path = Path(os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")))
    price_data_dir = root_path / "price_data"
    price_data_dir.mkdir(parents=True, exist_ok=True)

    vix = get_vix_data()
    vix_features = create_vix_features(vix)
    vix_features.to_parquet(price_data_dir / "vix_features.parquet")
