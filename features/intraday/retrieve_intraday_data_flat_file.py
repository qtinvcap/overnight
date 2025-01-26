import boto3
import gzip
import io
import pandas as pd
from botocore.config import Config
import os

###############################################################################
# 1) S3 CLIENT SETUP
###############################################################################

api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY environment variable is not set")


session = boto3.Session(
    aws_access_key_id="e3ef930d-f7ae-43ea-8b7c-219d7e1ea76f",
    aws_secret_access_key="QY2NsIqpUepPlZ0sedbd5LEBEHW5lQ35",
)

s3 = session.client(
    "s3",
    endpoint_url="https://files.polygon.io",  # Polygon's endpoint
    config=Config(signature_version="s3v4"),
)

bucket_name = "flatfiles"
prefix = "us_stocks_sip/minute_aggs_v1"  # Adjust if needed


###############################################################################
# 2) READ ONE DAY IN MEMORY + MARK 'market_statut'
###############################################################################
def fetch_one_day(date_obj):
    """
    Reads day_obj's .csv.gz in-memory from S3,
    adds 'market_statut' = ['open','intra-day','close','pre-market','after-market'].
    Returns a DataFrame or None if not found/empty.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    object_key = f"{prefix}/{year}/{month}/{date_str}.csv.gz"

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

    df["window_start"] = pd.to_datetime(df["window_start"], unit="ns").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

    df_filled = df

    # Create 'market_statut'
    df_filled["market_statut"] = pd.cut(
        df_filled["window_start"].dt.time,
        bins=[
            pd.to_datetime("00:00").time(),
            pd.to_datetime("09:30").time(),
            pd.to_datetime("16:00").time(),
            pd.to_datetime("23:59:59").time(),
        ],
        labels=["pre-market", "intra-day", "after-market"],
        include_lowest=True,
    ).astype(str)

    # Explicitly label 09:30 as 'open'
    df_filled.loc[df_filled["window_start"].dt.time == pd.to_datetime("09:30").time(), "market_statut"] = "open"
    # Explicitly label 16:00 as 'close'
    df_filled.loc[df_filled["window_start"].dt.time == pd.to_datetime("16:00").time(), "market_statut"] = "close"

    # Add a column with just the date (e.g. 2024-12-31) for clarity
    df_filled["trade_date"] = df_filled["window_start"].dt.date

    return df_filled
