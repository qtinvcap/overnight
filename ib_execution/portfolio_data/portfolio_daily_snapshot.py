import psycopg2
from datetime import datetime
import sys
from sqlalchemy import create_engine
from ib_execution.portfolio_data.retrieve_portfolio_data import fetch_portfolio_dataframe
from ib_execution.portfolio_data.utils import test_connect_to_ib, connect_to_ib
import subprocess
import logging
import time
import os
import pandas as pd

directory_path = os.environ["OVERNIGHT_ROOT_PATH"]

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

""" This script is mean to run once at the close to append the daily portfolio snapshot to the SQL table"""

# To connect to the sql table from the terminal:
# psql -h database-1.cu01gw6frkrx.eu-north-1.rds.amazonaws.com -U postgres -d postgres
# password: tradingrecord


def start_ibgateway():
    # Execute the bash script to restart IB Gateway
    command = f"nohup {directory_path}restart_ibgateway.sh > /home/ubuntu/trading_project/ibgateway.log 2>&1 &"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Running
    if result.returncode == 0:
        print("Bash script executed successfully.")
        print(result.stdout)
    else:
        print("Error executing bash script:", result.stderr)


def ensure_ib_connection():
    if test_connect_to_ib():
        return True
    else:
        logger.info("Server is currently not connected to IB Gateway: Attempting to restart IB Gateway...")
        start_ibgateway()
        time.sleep(20)  # Wait for 20 seconds before checking the connection again
        return test_connect_to_ib()


# Function to insert DataFrame into PostgreSQL
def insert_portfolio_snapshot(df):

    # Database connection parameters
    db_config = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "tradingrecord",
        "host": "database-1.cu01gw6frkrx.eu-north-1.rds.amazonaws.com",
    }

    # Create SQLAlchemy engine
    engine = create_engine(
        f'postgresql://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}/{db_config["dbname"]}'
    )
    try:

        # Add a snapshot_date column to the DataFrame
        df["snapshot_date"] = datetime.now().date()  # or the specific end of the day date
        existing_dates = pd.read_sql("SELECT DISTINCT snapshot_date FROM portfolio_snapshots", engine)

        # Filter out rows with existing dates
        df_to_insert = df[~df["snapshot_date"].isin(existing_dates["snapshot_date"])]

        if not df_to_insert.empty:
            df_to_insert.to_sql("portfolio_snapshots", engine, if_exists="append", index=False)
            logger.info(f"Inserted {len(df_to_insert)} new records.")

        else:
            logger.info("An entry was already existing for this snapshot date. No new records to insert.")
    except Exception as e:
        logger.error(f"Error inserting portfolio snapshot: {e}")


def connect_to_db():
    try:
        conn = psycopg2.connect(
            host="database-1.cu01gw6frkrx.eu-north-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="tradingrecord",
        )
        return conn
    except psycopg2.DatabaseError as e:
        print(f"Error: {e}")
        sys.exit(1)


def query_data(conn):
    cursor = conn.cursor()
    query_command = "SELECT * FROM portfolio_snapshots;"
    cursor.execute(query_command)
    records = cursor.fetchall()
    for record in records:
        print(record)
    cursor.close()


def main():
    max_retries = 3
    for attempt in range(max_retries):
        if ensure_ib_connection():
            try:
                ib = connect_to_ib()
                if ib is not None:
                    df = fetch_portfolio_dataframe(ib)
                    insert_portfolio_snapshot(df)
                    ib.disconnect()
                    break  # Success, exit the retry loop
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 30 seconds...")
                    time.sleep(30)
                else:
                    logger.error("All connection attempts failed.")
        else:
            logger.info("Failed to connect to IB Gateway.")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 30 seconds...")
                time.sleep(30)
            else:
                logger.error("All connection attempts failed.")


if __name__ == "__main__":
    main()
