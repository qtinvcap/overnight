"""
Backtest function to replicate the IB commission structure which can be found here:
https://www.interactivebrokers.com/en/pricing/commissions-stocks.php
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the IB tiered commission brackets: each tuple is (cumulative_limit, rate)
TIER_BRACKETS = [
    (300_000, 0.0035),
    (3_000_000, 0.0020),
    (20_000_000, 0.0015),
    (100_000_000, 0.0010),
    (float("inf"), 0.0005),
]


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


def calculate_ib_commission_cumulative(num_shares, price, cum_volume, pricing="tiered", side="entry"):
    """
    Calculate the commission for a single order leg (entry or exit)
    using cumulative monthly volume.

    For tiered pricing, shares are allocated across the volume tiers.
    A minimum commission of USD 0.35 applies and commission is capped at 1% of trade value.

    When side=='exit', extra fees are added:
      - Regulatory fees: SEC fee = 0.000008 × (num_shares×price)
                          FINRA fee = 0.000166 × num_shares
      - Exchange fees: 0.00295 per share
      - Clearing fees: 0.00020 per share
      - Pass-through fees: base commission × (0.000175+0.00056) = base commission×0.000735

    Returns a tuple (total_commission, new_cum_volume).
    """
    trade_value = num_shares * price
    remaining_shares = num_shares
    commission = 0.0

    if pricing == "fixed":
        rate = 0.005
        commission = num_shares * rate
        new_cum_volume = cum_volume + num_shares  # update volume if needed
    elif pricing == "tiered":
        new_cum_volume = cum_volume
        # Process the shares across tiers based on the current cumulative volume.
        for limit, rate in TIER_BRACKETS:
            if remaining_shares <= 0:
                break
            if new_cum_volume < limit:
                capacity = limit - new_cum_volume
                shares_in_bracket = min(remaining_shares, capacity)
                commission += shares_in_bracket * rate
                new_cum_volume += shares_in_bracket
                remaining_shares -= shares_in_bracket
        # In theory, any remaining shares go in the last bracket.
        if remaining_shares > 0:
            commission += remaining_shares * TIER_BRACKETS[-1][1]
            new_cum_volume += remaining_shares
    else:
        raise ValueError("pricing must be 'tiered' or 'fixed'")

    # Enforce minimum commission per order leg:
    if commission < 0.35:
        commission = 0.35

    # At this point commission is our base commission.
    base_commission = commission

    # For exit legs, add extra fees.
    if side == "exit":
        sec_fee = 0.000008 * (num_shares * price)
        finra_fee = 0.000166 * num_shares
        regulatory_fee = sec_fee + finra_fee
        exchange_fee = num_shares * 0.00295
        clearing_fee = num_shares * 0.00020
        pass_through_fee = base_commission * 0.000735  # (0.000175 + 0.00056)
        extra_fees = regulatory_fee + exchange_fee + clearing_fee + pass_through_fee
    else:
        extra_fees = 0.0

    total_commission = base_commission + extra_fees
    # Cap total commission at 1% of trade value:
    total_commission = min(total_commission, 0.01 * trade_value)
    return total_commission, new_cum_volume


def net_return_with_commission(
    raw_return, entry_price, position_size, cum_volume_entry, cum_volume_exit, pricing="tiered", slippage=0.0
):
    """
    Calculate net return after commissions and slippage using cumulative monthly volumes.
    The effective entry and exit prices are adjusted by slippage.
    Commissions are calculated separately on the entry leg (without extra fees)
    and the exit leg (with extra fees), and then subtracted from the position.

    Returns a tuple: (net_return, updated_cum_volume_entry, updated_cum_volume_exit)
    """
    num_shares = int(position_size / entry_price)
    if num_shares == 0:
        return 0.0, cum_volume_entry, cum_volume_exit

    # Apply slippage:
    adj_entry_price = entry_price * (1 + slippage)
    exit_price = entry_price * (1 + raw_return)
    adj_exit_price = exit_price * (1 - slippage)

    adjusted_raw_return = adj_exit_price / adj_entry_price - 1
    trade_value = num_shares * adj_entry_price

    # Compute commissions for entry and exit legs:
    commission_entry, new_cum_volume_entry = calculate_ib_commission_cumulative(
        num_shares, adj_entry_price, cum_volume_entry, pricing, side="entry"
    )
    commission_exit, new_cum_volume_exit = calculate_ib_commission_cumulative(
        num_shares, adj_exit_price, cum_volume_exit, pricing, side="exit"
    )
    total_commission = commission_entry + commission_exit

    net_return = (position_size * (1 + adjusted_raw_return) - total_commission) / position_size - 1
    return net_return, new_cum_volume_entry, new_cum_volume_exit


def backtest_with_commission(
    df,
    threshold=0.4,
    max_positions=10,
    initial_capital=50000,
    date_col="trade_date",
    non_adjusted_close_colum="intraday_last_close_before_1555",
    adjusted_close_column="orig_close",
    adjusted_next_open_column="next_day_open",
    probas_col="total_accuracy_score",
    liquidity_threshold=10000,
    price_threshold=1,
    pricing="tiered",  # 'tiered', 'fixed', or 'lite'
    slippage=0.0,
):
    """
    Backtest strategy with IB commission structure that accounts for cumulative monthly volume.
    We maintain separate monthly cumulative volumes for the entry and exit legs.
    """
    monthly_volume_entry = {}
    monthly_volume_exit = {}

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    grouped = df_sorted.groupby(date_col, as_index=False)

    daily_records = []
    positions_list = []
    capital = initial_capital
    unique_dates = df_sorted[date_col].unique()

    for day in unique_dates:
        month_key = pd.to_datetime(day).strftime("%Y-%m")
        if month_key not in monthly_volume_entry:
            monthly_volume_entry[month_key] = 0
            monthly_volume_exit[month_key] = 0

        day_data = grouped.get_group(day)

        qualified = day_data[
            (day_data["roll5_mean_intraday_total_dollar_volume_all"] > liquidity_threshold)
            & (day_data[non_adjusted_close_colum] > price_threshold)
        ].copy()

        qualified.sort_values(
            by=[probas_col, "roll5_mean_intraday_total_dollar_volume_all"], ascending=[False, False], inplace=True
        )

        qualified = qualified[qualified[probas_col] >= threshold].dropna(subset=[non_adjusted_close_colum, probas_col])

        if len(qualified) == 0:
            daily_return = 0.0
            n_positions = 0
        else:
            topN = qualified.head(max_positions)
            n_positions = len(topN)
            position_size = capital / n_positions

            net_returns = []
            for _, row in topN.iterrows():
                raw_return = row[adjusted_next_open_column] / row[adjusted_close_column] - 1.0

                current_entry_volume = monthly_volume_entry[month_key]
                current_exit_volume = monthly_volume_exit[month_key]

                net_ret, new_entry_volume, new_exit_volume = net_return_with_commission(
                    raw_return,
                    row[non_adjusted_close_colum],
                    position_size,
                    current_entry_volume,
                    current_exit_volume,
                    pricing=pricing,
                    slippage=slippage,
                )
                net_returns.append(net_ret)
                monthly_volume_entry[month_key] = new_entry_volume
                monthly_volume_exit[month_key] = new_exit_volume

                num_shares = int(position_size / row[non_adjusted_close_colum])
                slippage_adjusted_entry_price = row[non_adjusted_close_colum] * (1 + slippage)
                slippage_adjusted_exit_price = row[non_adjusted_close_colum] * (
                    1 - slippage
                )  #### ASSUMING EXIT PRICE IS THE SAME AS ENTRY PRICE (NON ADJUSTED FOR STOCK SPLITS)

                commission_entry, _ = calculate_ib_commission_cumulative(
                    num_shares, slippage_adjusted_entry_price, current_entry_volume, pricing, side="entry"
                )
                commission_exit, _ = calculate_ib_commission_cumulative(
                    num_shares, slippage_adjusted_exit_price, current_exit_volume, pricing, side="exit"
                )
                commission_paid = commission_entry + commission_exit
                commission_pct = commission_paid / position_size * 100

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
                    "commission_pct": commission_pct,
                    "commission_note": "",
                }
                positions_list.append(position_info)

            net_returns = np.array(net_returns)
            daily_return = net_returns.mean()

        capital *= 1.0 + daily_return

        daily_records.append(
            {"date": day, "daily_return": daily_return, "capital": capital, "n_positions": n_positions}
        )

    daily_df = pd.DataFrame(daily_records).sort_values("date").set_index("date")
    positions_df = pd.DataFrame(positions_list)
    if not positions_df.empty:
        positions_df["win"] = positions_df["net_return"] > 0

    daily_df["cum_return"] = daily_df["capital"] / initial_capital - 1.0
    running_max = daily_df["capital"].cummax()
    daily_df["drawdown"] = 1 - (daily_df["capital"] / running_max)
    max_dd = daily_df["drawdown"].max()

    total_days = len(daily_df)
    annual_return = (1 + daily_df["cum_return"].iloc[-1]) ** (252 / total_days) - 1 if total_days >= 2 else np.nan
    daily_return_std = daily_df["daily_return"].std()
    sharpe = (daily_df["daily_return"].mean() / daily_return_std * np.sqrt(252)) if daily_return_std != 0 else np.nan
    annual_vol = daily_return_std * np.sqrt(252)
    mar_ratio = annual_return / max_dd if max_dd > 0 else np.nan

    var_1 = np.percentile(daily_df["daily_return"], 1)
    var_5 = np.percentile(daily_df["daily_return"], 5)
    var_10 = np.percentile(daily_df["daily_return"], 10)

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
        "avg_nb_of_position_with_price_under_1": len(positions_df[positions_df["non_adjusted_entry_price"] < 1])
        / len(positions_df),
    }

    return daily_df, stats, positions_df


# Example usage:
# Ensure your DataFrame 'global_scored_df' contains the required columns.
daily_results, performance, positions_df = backtest_with_commission(
    df=global_scored_df.dropna(subset=["total_accuracy_score"]),
    threshold=0.5,
    max_positions=5,
    initial_capital=50000,
    date_col="trade_date",
    non_adjusted_close_colum="intraday_last_close_before_1555",
    adjusted_close_column="orig_close",
    adjusted_next_open_column="next_day_open",
    probas_col="total_accuracy_score",
    liquidity_threshold=2000000,
    price_threshold=1,
    pricing="tiered",  # Options: 'tiered' or 'fixed'
    slippage=0.0001,
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

    print("\nTop 5 Performing Tickers:")
    ticker_stats = (
        positions_df.groupby("ticker")
        .agg({"net_return": ["count", "mean", "std"], "probability": "mean", "commission": "mean"})
        .round(4)
    )
    print(ticker_stats.sort_values(("net_return", "mean"), ascending=False).head())

# Plot equity curve and drawdown
daily_results[["capital"]].plot(figsize=(10, 5), title="Equity Curve")
plt.show()

daily_results[["drawdown"]].plot(figsize=(10, 3), title="Drawdown")
plt.show()
