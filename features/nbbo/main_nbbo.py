from features.nbbo.features_engineering import compute_nbbo_features
from features.nbbo.retrieve_nbbo_data import fetch_quotes_in_window
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from datetime import datetime
from tqdm import tqdm


def gather_nbbo_features_over_date_range(ticker, start_date_str, end_date_str):
    """
    Iterate over all valid NYSE trading days between start_date_str and end_date_str,
    fetch NBBO quotes for two windows:
        1) 14:55–15:55 (ET) to compute 'last-hour' features
        2) 15:55–16:00 (ET) for closing auction liquidity
    Return a DataFrame with daily metrics for both windows.

    :param ticker: e.g. "AAPL"
    :param start_date_str: e.g. "2024-01-01"
    :param end_date_str: e.g. "2024-01-31"
    :return: pd.DataFrame with columns for each metric, plus date
    """
    # 1) Build a schedule of valid trading days
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)

    eastern_tz = pytz.timezone("US/Eastern")

    results = []

    for current_day in tqdm(schedule.index, desc=f"Processing {ticker}"):
        # current_day is a pandas Timestamp (midnight in UTC), convert to date
        day_date = current_day.date()

        # Construct local datetimes in ET
        # Window 1: 14:55–15:55 ET
        start_local_1 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 14, 55, 0))
        end_local_1 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 15, 55, 0))

        # Window 2: 15:55–16:00 ET
        start_local_2 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 15, 55, 0))
        end_local_2 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 16, 0, 0))

        # 2) Fetch quotes & compute features for each window
        quotes_1 = fetch_quotes_in_window(ticker, start_local_1, end_local_1)
        feat_1 = compute_nbbo_features(quotes_1)  # last hour window

        quotes_2 = fetch_quotes_in_window(ticker, start_local_2, end_local_2)
        feat_2 = compute_nbbo_features(quotes_2)  # final 5 min window

        # 3) Combine features into a single row/dict for the day
        # Prefix keys so we can distinguish them easily
        row = {
            "date": day_date,
        }
        for k, v in feat_1.items():
            row[f"lh_{k}"] = v  # "lh_" = last hour
        for k, v in feat_2.items():
            row[f"ca_{k}"] = v  # "ca_" = closing auction

        results.append(row)

    # 4) Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results


###############################################################################
# 6) DEMO USAGE IN MAIN
###############################################################################
def main():
    # Example usage:
    ticker = "TSLA"
    start_date_str = "2024-06-01"
    end_date_str = "2024-06-10"

    df_nbbo = gather_nbbo_features_over_date_range(ticker, start_date_str, end_date_str)
    print(df_nbbo.head(10))

    return df_nbbo


if __name__ == "__main__":
    df_nbbo = main()
    print(df_nbbo.head(10))
