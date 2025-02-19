def analyze_annual_performance(daily_results):
    """
    Analyze and plot annual performance metrics from backtest results
    """
    # Add year to daily_df if not already present
    daily_results = daily_results.copy()
    daily_results["year"] = daily_results.index.year

    # Calculate annual performance
    annual_performance = (
        daily_results.groupby("year")
        .apply(
            lambda x: {
                "return": (1 + x["daily_return"]).prod() - 1,
                "sharpe": x["daily_return"].mean() / x["daily_return"].std() * np.sqrt(252) if len(x) > 1 else 0,
                "max_drawdown": x["drawdown"].max(),
                "avg_positions": x["n_positions"].mean(),
            }
        )
        .apply(pd.Series)
    )

    # Plot annual performance
    plt.figure(figsize=(12, 6))
    annual_performance["return"].plot(kind="bar")
    plt.title("Annual Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add return values on top of bars
    for i, v in enumerate(annual_performance["return"]):
        plt.text(i, v, f"{v:.1%}", ha="center", va="bottom" if v > 0 else "top")

    plt.tight_layout()
    plt.show()

    return annual_performance


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_momentum_ranking(df, liquidity_threshold, price_threshold):
    """
    Calculate 5-day close-to-close performance and rank tickers by this performance for each day.
    Rank 1 = highest performance.
    """
    df = df[df["roll5_mean_intraday_total_dollar_volume_all"] > liquidity_threshold]
    df = df[df["orig_close"] > price_threshold]

    # Calculate 5-day close-to-close performance
    df = df.sort_values(["ticker", "trade_date"]).copy()
    df["close_5d_ago"] = df.groupby("ticker")["close"].shift(5)
    df["perf_5d"] = df["close"] / df["close_5d_ago"] - 1

    # Rank tickers by performance for each day (1 = highest performance)
    df["momentum_ranking"] = df.groupby("trade_date")["perf_5d"].rank(ascending=False)

    # Clean up intermediate columns
    df.drop(["close_5d_ago", "perf_5d"], axis=1, inplace=True)

    return df


def backtest_with_commission(
    df,
    threshold=0.35,
    max_positions=10,
    initial_capital=50000,
    date_col="trade_date",
    non_adjusted_close_colum="intraday_last_close_before_1555",
    adjusted_close_column="orig_close",
    adjusted_next_open_column="next_day_open",
    probas_col="total_accuracy_score",
    liquidity_threshold=10000,
    price_threshold=1,
    top_n_momentum=30,
    com_per_share=0.0035,
    slippage=0.0,  # Optional slippage as a decimal (e.g., 0.001 = 0.1%)
):
    """
    Backtest strategy with IB commission structure:
      - USD 0.0035 per share commission
      - Total commission capped at 1% of trade value (i.e. number_of_shares * effective_entry_price)

    Capital is updated daily, and commission for each trade is recorded as a percentage of the allocated position size.
    """
    # Apply momentum ranking filter if desired
    if top_n_momentum is not None:
        df = add_momentum_ranking(df, liquidity_threshold, price_threshold)
        df = df[df["momentum_ranking"] < top_n_momentum]

    # Sort the DataFrame by trading date
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    grouped = df_sorted.groupby(date_col, as_index=False)

    # Initialize tracking variables
    daily_records = []
    positions_list = []
    capital = initial_capital
    unique_dates = df_sorted[date_col].unique()

    def net_return_with_commission(raw_return, entry_price, position_size):
        """
        Calculate net return after commissions and slippage:
          - Effective entry price = entry_price * (1 + slippage)
          - Effective exit price  = exit_price * (1 - slippage)
            where exit_price is derived as entry_price*(1 + raw_return).
          - Commission: USD 0.0035 per share, capped at 1% of trade value (based on effective entry price)
        """
        # Determine number of shares (round down)
        num_shares = int(position_size / entry_price)
        if num_shares == 0:
            return 0.0

        # Compute effective prices with slippage
        slippage_adjusted_entry_price = entry_price * (1 + slippage)
        # Derive the exit price from raw_return and then apply slippage on exit
        exit_price = entry_price * (1 + raw_return)
        slippage_adjusted_exit_price = exit_price * (1 - slippage)

        # Adjusted raw return with slippage factored in
        adjusted_raw_return = slippage_adjusted_exit_price / slippage_adjusted_entry_price - 1

        # Commission calculations
        commission_per_share = com_per_share
        trade_value = num_shares * slippage_adjusted_entry_price
        total_commission = min(2 * num_shares * commission_per_share, trade_value * 0.01)

        net_return = (position_size * (1 + adjusted_raw_return) - total_commission) / position_size - 1

        return net_return

    # Loop over each trading day
    for day in unique_dates:
        day_data = grouped.get_group(day)

        # Filter for liquidity and price thresholds
        qualified = day_data[
            (day_data["roll5_mean_intraday_total_dollar_volume_all"] > liquidity_threshold)
            & (day_data[non_adjusted_close_colum] > price_threshold)
        ].copy()

        # Sort by probability score and liquidity (both descending)
        qualified.sort_values(
            by=[probas_col, "roll5_mean_intraday_total_dollar_volume_all"], ascending=[False, False], inplace=True
        )

        # Filter by probability threshold and drop rows with missing data
        qualified = qualified[qualified[probas_col] >= threshold].dropna(subset=[non_adjusted_close_colum, probas_col])

        if len(qualified) == 0:
            daily_return = 0.0
            n_positions = 0
        else:
            # Select top N positions (max_positions)
            topN = qualified.head(max_positions)
            n_positions = len(topN)
            # Use current capital for realistic compounding position sizing
            position_size = capital / n_positions

            net_returns = []
            for _, row in topN.iterrows():
                raw_return = row[adjusted_next_open_column] / row[adjusted_close_column] - 1.0
                net_ret = net_return_with_commission(raw_return, row[non_adjusted_close_colum], position_size)
                net_returns.append(net_ret)

            net_returns = np.array(net_returns)
            daily_return = net_returns.mean()

            # Record individual trade details
            for _, row in topN.iterrows():
                raw_return = row[adjusted_next_open_column] / row[adjusted_close_column] - 1.0
                net_ret = net_return_with_commission(raw_return, row[non_adjusted_close_colum], position_size)

                # Determine number of shares based on the non-slippage price (for consistency)
                num_shares = int(position_size / row[non_adjusted_close_colum])
                # Apply slippage to get the effective entry price
                splippage_adjusted_entry_price = row[non_adjusted_close_colum] * (1 + slippage)
                trade_value = num_shares * splippage_adjusted_entry_price

                # Commission parameters
                commission_per_share = com_per_share
                # Total commission computed on both legs, capped at 1% of the trade value
                commission_cap = trade_value * 0.01
                commission_paid = min(2 * num_shares * commission_per_share, commission_cap)
                # Calculate commission as a percentage of the allocated position size
                commission_pct = commission_paid / position_size * 100
                # Note if the commission cap was applied
                commission_note = "Commission cap of 1% applied" if commission_paid == commission_cap else ""

                position_info = {
                    "date": day,
                    "ticker": row["ticker"],
                    "non_adjusted_entry_price": row[non_adjusted_close_colum],
                    "adjusted_entry_price": row[adjusted_close_column],
                    "adjusted_exit_price": row[adjusted_next_open_column],
                    "raw_return": raw_return,
                    "net_return": net_ret,
                    "probability": row[probas_col],
                    "volume": row["roll5_mean_intraday_total_dollar_volume_all"],
                    "position_size": position_size,
                    "num_shares": num_shares,
                    "commission": commission_paid,
                    "commission_pct": commission_pct,  # Commission as a percentage of position size
                    "commission_note": commission_note,
                }
                positions_list.append(position_info)

        # Update capital using the compounded daily return
        capital = capital * (1.0 + daily_return)

        # Record daily summary data
        daily_records.append(
            {"date": day, "daily_return": daily_return, "capital": capital, "n_positions": n_positions}
        )

    # Build DataFrames for daily performance and individual trades
    daily_df = pd.DataFrame(daily_records)
    daily_df.sort_values("date", inplace=True)
    daily_df.set_index("date", inplace=True)

    positions_df = pd.DataFrame(positions_list)
    if not positions_df.empty:
        positions_df["win"] = positions_df["net_return"] > 0

    # Compute cumulative return and drawdown statistics
    daily_df["cum_return"] = daily_df["capital"] / initial_capital - 1.0
    running_max = daily_df["capital"].cummax()
    daily_df["drawdown"] = 1 - (daily_df["capital"] / running_max)
    max_dd = daily_df["drawdown"].max()

    # Annualized metrics
    total_days = len(daily_df)
    if total_days < 2:
        annual_return = np.nan
    else:
        final_cum_return = daily_df["cum_return"].iloc[-1]
        annual_return = (1 + final_cum_return) ** (252 / total_days) - 1

    daily_return_std = daily_df["daily_return"].std()
    daily_return_mean = daily_df["daily_return"].mean()
    sharpe = (daily_return_mean / daily_return_std * np.sqrt(252)) if daily_return_std != 0 else np.nan
    annual_vol = daily_return_std * np.sqrt(252)
    mar_ratio = annual_return / max_dd if max_dd > 0 else np.nan

    # Calculate Value at Risk (VaR)
    var_1 = np.percentile(daily_df["daily_return"], 1)
    var_5 = np.percentile(daily_df["daily_return"], 5)
    var_10 = np.percentile(daily_df["daily_return"], 10)

    # Compile performance statistics
    stats = {
        "final_capital": capital,
        "final_cum_return": daily_df["cum_return"].iloc[-1],
        "max_drawdown": max_dd,
        "annualized_return": annual_return,
        "mar_ratio": mar_ratio,
        "annualized_volatility": annual_vol,
        "var_1": var_1,
        "var_5": var_5,
        "var_10": var_10,
        "sharpe_ratio": sharpe,
        "total_trades": len(positions_df),
        "win_rate": positions_df["win"].mean() if not positions_df.empty else 0,
        "avg_return_per_trade": positions_df["net_return"].mean() if not positions_df.empty else 0,
        "avg_positions_per_day": daily_df["n_positions"].mean(),
        "avg_commission_pct_per_trade": positions_df["commission_pct"].mean() if not positions_df.empty else 0,
    }

    return daily_df, stats, positions_df


daily_results, performance, positions_df = backtest_with_commission(
    df=global_scored_df.dropna(subset=["total_accuracy_score"]),
    threshold=0.4,
    max_positions=5,
    initial_capital=50000,
    date_col="trade_date",
    non_adjusted_close_colum="intraday_last_close_before_1555",
    adjusted_close_column="orig_close",
    adjusted_next_open_column="next_day_open",
    probas_col="total_accuracy_score",
    liquidity_threshold=1000000,
    top_n_momentum=None,
    slippage=0.0001,
    price_threshold=1,
    com_per_share=0.0035,
)

annual_performance = analyze_annual_performance(daily_results)

# Print performance summary
print("\nPerformance Stats:")
for k, v in performance.items():
    print(f"{k}: {v:.4f}")

# Print position analysis
if len(positions_df) > 0:
    print("\nPosition Statistics:")
    print(f"Total Trades: {len(positions_df)}")
    print(f"Win Rate: {positions_df['win'].mean():.2%}")
    print(f"Average Return per Trade: {positions_df['net_return'].mean():.2%}")
    print(f"Average Commission per Trade: ${positions_df['commission'].mean():.4f}")

    # Top performing tickers
    print("\nTop 5 Performing Tickers:")
    ticker_stats = (
        positions_df.groupby("ticker")
        .agg({"net_return": ["count", "mean", "std"], "probability": "mean", "commission": "mean"})
        .round(4)
    )
    print(ticker_stats.sort_values(("net_return", "mean"), ascending=False).head())

# Plot equity curve
daily_results[["capital"]].plot(figsize=(10, 5), title="Equity Curve")
plt.show()

# Plot drawdown
daily_results[["drawdown"]].plot(figsize=(10, 3), title="Drawdown")
plt.show()
