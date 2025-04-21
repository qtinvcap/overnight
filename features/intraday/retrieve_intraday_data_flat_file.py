import boto3
import gzip
import io
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ResponseStreamingError, BotoCoreError
import os
import time
import logging

###############################################################################
# 1) S3 CLIENT SETUP
###############################################################################

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY environment variable is not set")

session = boto3.Session(
    aws_access_key_id="e3ef930d-f7ae-43ea-8b7c-219d7e1ea76f",
    aws_secret_access_key="QY2NsIqpUepPlZ0sedbd5LEBEHW5lQ35",
)

# Increase timeouts in the botocore config
s3_config = Config(
    signature_version="s3v4",
    connect_timeout=60,
    read_timeout=300
)

s3 = session.client(
    "s3",
    endpoint_url="https://files.polygon.io",  # Polygon's endpoint
    config=s3_config,
)

bucket_name = "flatfiles"
prefix = "us_stocks_sip/minute_aggs_v1"  # Adjust if needed


###############################################################################
# 2) READ ONE DAY IN MEMORY + MARK 'market_statut'
###############################################################################
def fetch_one_day(date_obj, max_retries=5, initial_backoff=1):
    """
    Reads day_obj's .csv.gz in-memory from S3 with retries,
    adds 'market_statut' = ['open','intra-day','close','pre-market','after-market'].
    Returns a DataFrame or None if not found/empty or after max retries.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    object_key = f"{prefix}/{year}/{month}/{date_str}.csv.gz"

    retries = 0
    backoff = initial_backoff
    while retries < max_retries:
        try:
            # Fetch from S3
            logger.info(f"Attempting to fetch {object_key} (Retry {retries + 1}/{max_retries})")
            response = s3.get_object(Bucket=bucket_name, Key=object_key)

            # Read the body (this is where the IncompleteRead happened)
            logger.info(f"Reading response body for {object_key}")
            gz_body = response["Body"].read()
            logger.info(f"Successfully read {len(gz_body)} bytes for {object_key}")

            # Process the data
            with gzip.GzipFile(fileobj=io.BytesIO(gz_body)) as gz_file:
                df = pd.read_csv(gz_file, low_memory=False)
            if df.empty:
                logger.warning(f"Data found but empty for {object_key}")
                return None

            # If successful, break the loop and proceed
            logger.info(f"Successfully processed {object_key}")
            break

        except (ResponseStreamingError, BotoCoreError, OSError) as e:
            retries += 1
            logger.warning(f"Error fetching/reading {object_key} (Retry {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                logger.error(f"Max retries reached for {object_key}. Failing.")
                return None
            
            logger.info(f"Waiting {backoff} seconds before retrying...")
            time.sleep(backoff)
            backoff *= 2

        except Exception as e:
            if "NoSuchKey" in str(e):
                 logger.warning(f"Missing data (NoSuchKey) for {object_key}. No retry needed.")
            else:
                logger.error(f"Non-retryable error fetching {object_key}: {e}")
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
