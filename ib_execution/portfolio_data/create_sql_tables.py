import psycopg2
import sys

""" script to create a table in PostgreSQL database. this won't be used once initial table is created"""


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


def delete_table(conn):
    cursor = conn.cursor()
    delete_table_command = "DROP TABLE IF EXISTS portfolio_snapshots;"
    cursor.execute(delete_table_command)
    conn.commit()
    cursor.close()


def create_table(conn):
    cursor = conn.cursor()
    create_table_command = """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        snapshot_date DATE NOT NULL,
        ticker VARCHAR(10),
        position_size INT,
        market_value NUMERIC(15,2),
        average_cost NUMERIC(15,2),
        latest_price NUMERIC(15,2),
        unrealized_pnl NUMERIC(15,2),
        net_liquidation NUMERIC(15,2) NOT NULL,
        total_cash_value NUMERIC(15,2) NOT NULL
    );
    """
    cursor.execute(create_table_command)
    conn.commit()
    cursor.close()


if __name__ == "__main__":
    conn = connect_to_db()
    #delete_table(conn)
    create_table(conn)
    conn.close()
