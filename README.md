# Overnight - Automated Intraday Stock Trading System

An end-to-end automated trading system that streams real-time market data, engineers features from intraday and daily price action, scores stocks using ensemble ML models, and executes trades via Interactive Brokers -- all running autonomously during market hours.

## Architecture

```
                    +-----------------------+
                    |    Polygon WebSocket   |
                    |  (1-min bars, all US   |
                    |       equities)        |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Streaming Processor  |
                    |  Collects bars 4AM-8PM |
                    |  Time-triggered @15:56 |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v---------+             +-----------v-----------+
    |  Intraday Features |             |    Daily Features     |
    |                    |             |                       |
    | - Return windows   |             | - RSI, MACD, ATR      |
    | - Volatility       |             | - Bollinger Bands     |
    | - VWAP deviation   |             | - Pivot Points        |
    | - Dollar volume    |             | - Rolling statistics  |
    | - Price accel.     |             | - ETF relative stats  |
    | - ETF correlation  |             | - Candlestick flags   |
    +---------+----------+             +-----------+-----------+
              |                                   |
              +-----------------+-----------------+
                                |
                    +-----------v-----------+
                    |   Feature Combiner     |
                    |  Z-scores, ratios,     |
                    |  temporal features     |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Model Ensemble (v3)  |
                    |                        |
                    | 3 models with          |
                    | decision-tree-based    |
                    | feature range scoring  |
                    | + linear regression    |
                    | weighting              |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |    Signal Generator    |
                    |  Liquidity + price     |
                    |  filters, top-5 picks  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Interactive Brokers    |
                    |  Close Price algo entry |
                    |  OPG + DAY exit orders  |
                    +------------------------+
```

## How It Works

The system exploits overnight return patterns in small/mid-cap US equities:

1. **Data Collection** -- WebSocket connection to Polygon streams 1-minute bars for the entire US equity market from pre-market (4 AM ET) through after-hours (8 PM ET).

2. **Feature Engineering** -- At market close (~15:56 ET), the system computes ~150+ features per stock across two axes:
   - **Intraday**: return windows, volatility, dollar volume profiles, VWAP deviation, price acceleration
   - **Daily**: technical indicators (RSI, MACD, Bollinger, ATR, OBV), rolling statistics, ETF-relative metrics (vs SPY, QQQ, IWM)
   - Features are z-score normalized using 20-day rolling statistics

3. **Scoring** -- A three-model ensemble scores each stock:
   - Decision trees partition feature space into optimal ranges
   - Linear regression weights range-membership signals
   - Ensemble combines models targeting different performance quantiles (top 0.5%, top/bottom 2%)

4. **Execution** -- Trades are routed through Interactive Brokers:
   - **Entry**: Close Price algorithm orders at ~16:00 ET (minimizes market impact)
   - **Exit**: OPG orders at next-day open (9:29 AM), with DAY fallback for unfilled positions
   - Position sizing: equal-weight across up to 5 positions

## Daily Schedule (ET)

| Time | Action |
|------|--------|
| 03:55 | Start WebSocket streaming |
| 09:28 | Cancel pending premarket orders |
| 09:29 | Place OPG exit orders for overnight positions |
| 09:30 | Place DAY exit orders for remaining positions |
| 10:05 | Prepare daily features (205-day lookback) |
| 15:56 | Compute intraday features, run model inference |
| 16:00 | Monitor fills, place premarket limit entries |
| 20:00 | End of session |

## Project Structure

```
overnight/
├── pipeline/                       # Orchestration
│   ├── main_trading_pipeline_with_ib.py   # Production pipeline (daily schedule)
│   ├── streaming_and_process_intraday_data_time_trigger.py  # WebSocket streamer
│   └── backtest_pipeline.py        # Historical backtesting
│
├── features/                       # Feature engineering
│   ├── daily/                      # Daily technical indicators
│   │   ├── features_engineering_with_etf.py  # 50+ daily features w/ ETF comparisons
│   │   └── retrieve_price_data_api.py        # Polygon REST API (parallelized)
│   ├── intraday/                   # Intraday microstructure features
│   │   ├── features_engineering_with_etf.py  # Return windows, vol, VWAP, accel
│   │   ├── rolling_calcs.py        # 20-day rolling normalization
│   │   └── retrieve_intraday_data_flat_file.py  # S3 minute-bar reader
│   ├── nbbo/                       # National Best Bid/Offer features
│   ├── combine_intraday_daily_features.py    # Merge + normalize
│   └── vix.py                      # VIX regime features
│
├── models/                         # ML scoring models
│   ├── model_v3/                   # Production model (3-model ensemble)
│   │   ├── bin_search.py           # Decision-tree range optimization
│   │   └── main_inference.py       # Scoring + signal generation
│   ├── model_v2/                   # Previous iteration
│   └── model_v1/                   # Initial prototype
│
├── ib_execution/                   # Interactive Brokers integration
│   ├── orders_management/          # Order lifecycle (entry, exit, cancel)
│   └── portfolio_data/             # Cash, P&L, execution analysis
│
└── backtest/                       # Backtesting framework
    ├── performance_backtest.py     # Commission model + performance metrics
    └── create_dataset/             # Historical dataset construction
```

## Tech Stack

- **Data**: Polygon.io (WebSocket + REST), AWS S3, yfinance, NASDAQ Data Link
- **Compute**: pandas, NumPy, scikit-learn, multiprocessing
- **Execution**: Interactive Brokers via ib-insync
- **Infrastructure**: WebSocket streaming, threading, pyarrow (Parquet I/O)

## Setup

```bash
# Clone the repository
git clone https://github.com/<username>/overnight.git
cd overnight

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
export POLYGON_API_KEY="your_polygon_api_key"
export NASDAQDATALINK_API_KEY="your_nasdaq_data_link_key"
export OVERNIGHT_ROOT_PATH="/path/to/data/directory"

# Run the trading pipeline
python -m pipeline.main_trading_pipeline_with_ib
```

### Prerequisites

- **Polygon.io** account with WebSocket access (real-time data)
- **Interactive Brokers** account with TWS/IB Gateway running on port 4002
- **AWS credentials** configured for S3 access (historical minute bars)

## Backtesting

Run a historical backtest for any past trading day:

```python
from pipeline.backtest_pipeline import run_backtest

results = run_backtest("2025-01-08")
```

The backtest uses the same feature pipeline and model inference as production, with a realistic commission model (IB tiered pricing + SEC/FINRA fees).

## Model Evolution

| Version | Approach | Key Change |
|---------|----------|------------|
| v1 | Single feature-range scoring | Baseline |
| v2 | Linear regression weighting | Added LR on range membership |
| v3 | Three-model ensemble + ETF features | Multiple quantile targets, ETF-relative signals |
