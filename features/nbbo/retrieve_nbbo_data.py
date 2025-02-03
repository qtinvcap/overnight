import boto3
import gzip
import io
import pandas as pd
from datetime import datetime, timedelta
from botocore.config import Config
import pandas_market_calendars as mcal
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import numpy as np
import requests
import pytz
import os

###############################################################################
# 1) S3 CLIENT SETUP (if you need S3 data)
###############################################################################
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY environment variable is not set")


session = boto3.Session(
    aws_access_key_id="0b8566f5-0f1b-41b7-b947-af9dcdea0a87",
    aws_secret_access_key=api_key,
)

s3 = session.client(
    "s3",
    endpoint_url="https://files.polygon.io",  # Polygon's endpoint
    config=Config(signature_version="s3v4"),
)

bucket_name = "flatfiles"
prefix = "us_stocks_sip/quotes_v1"  # Adjust if needed

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io/v3/quotes"


###############################################################################
# 2) READ ONE DAY IN MEMORY FROM S3 + MARK 'market_statut' (Optional)
###############################################################################
def read_and_label_one_day(date_obj):
    """
    Reads date_obj's .csv.gz in-memory from S3,
    adds 'market_statut' = ['open','intra-day','close','pre-market','after-market'] (if you wish).
    Returns a DataFrame or None if not found/empty.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    object_key = f"{prefix}/{year}/{month}/{date_str}.csv.gz"
    print(object_key)

    # Fetch from S3
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
    except Exception as e:
        print(f"Missing data for {object_key}: {e}")
        return None

    gz_body = response["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(gz_body)) as gz_file:
        df = pd.read_csv(gz_file, low_memory=False)
    if df.empty:
        return None

    # You can optionally label the 'market_statut' here, if needed
    return df


###############################################################################
# 3) POLYGON QUOTES FETCHING UTILS
###############################################################################


def get_utc_timestamp(dt):
    """
    Convert a naive or local datetime to UTC in ISO 8601 string format.
    Polygon can handle query params in ISO 8601 UTC for timestamp.gte/lte.
    """
    dt_utc = dt.astimezone(pytz.UTC)
    return dt_utc.isoformat()


def fetch_quotes_in_window(ticker, start_dt_local, end_dt_local):
    """
    Fetch NBBO quotes from Polygon in the time window [start_dt_local, end_dt_local].
    start_dt_local and end_dt_local are Python datetime objects in Eastern Time,
    converted to UTC before querying Polygon.
    """
    start_iso_utc = get_utc_timestamp(start_dt_local)
    end_iso_utc = get_utc_timestamp(end_dt_local)

    params = {
        "timestamp.gte": start_iso_utc,
        "timestamp.lte": end_iso_utc,
        "limit": 50000,  # max items per response page
        "apiKey": API_KEY,
    }

    all_quotes = []
    url = f"{BASE_URL}/{ticker}"

    while True:
        print(f"Fetching quotes from {url} with window {start_iso_utc} - {end_iso_utc}")
        response = requests.get(url, params=params)
        data = response.json()

        if "results" in data:
            all_quotes.extend(data["results"])

        # If there's a 'next_url', keep fetching the next page
        if "next_url" in data and data["next_url"]:
            url = data["next_url"]
            # 'next_url' already includes params & apiKey, so set params={}
            params = {}
        else:
            break

    return all_quotes
