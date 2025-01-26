import requests
import pandas as pd
import pandas_market_calendars as mcal
from multiprocessing import Pool, cpu_count
import os


class PolygonHistoricalDailyData:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

    def get_daily_data(self, date):
        url = f"{self.base_url}/{date}"
        params = {"adjusted": "true", "include_otc": "false", "apiKey": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "OK":
                raise Exception(f"API returned status: {data['status']}")

            if not data.get("results"):
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")

            df = df.rename(
                columns={
                    "T": "ticker",
                    "c": "close",
                    "h": "high",
                    "l": "low",
                    "n": "transactions",
                    "o": "open",
                    "v": "volume",
                    "vw": "vwap",
                }
            )

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_historical_data(self, start_date_str, end_date_str, max_workers=None):
        """
        Retrieve historical daily data for a date range using parallel processing
        Returns a single DataFrame with all the data

        Args:
            start_date_str: Start date in YYYY-MM-DD format
            end_date_str: End date in YYYY-MM-DD format
            max_workers: Maximum number of worker processes (defaults to CPU count)
        """
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)
        trading_days = mcal.date_range(schedule, frequency="1D")
        trading_days = [d.date().strftime("%Y-%m-%d") for d in trading_days]

        # Use all available CPUs if max_workers not specified
        max_workers = max_workers or cpu_count()

        # Create a pool of workers
        with Pool(processes=max_workers) as pool:
            # Map the get_daily_data function to all trading days
            all_data = pool.map(self.get_daily_data, trading_days)

        # Filter out empty DataFrames and concatenate results
        all_data = [df for df in all_data if not df.empty]
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


if __name__ == "__main__":
    api_key = os.getenv("POLYGON_API_KEY")
    polygon = PolygonHistoricalDailyData(api_key)
    df = polygon.get_historical_data("2024-10-01", "2025-01-16")
    print(df)
