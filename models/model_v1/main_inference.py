import pandas as pd
from overnight.models.model_v1.bin_search import create_scoring_features, reformat_scored_df
import pickle


def load_best_ranges(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def score_features_df(
    df: pd.DataFrame,
    best_ranges: dict,
    metric: str = "trimmed_mean_std_ratio_performance",
    target_total_rules: int = 160,
    target_ratio: float = 0.5,
    weight_by_spread: bool = True,
) -> pd.DataFrame:
    scored_df, binary_features, rules_info = create_scoring_features(
        df=df,
        best_ranges=best_ranges,
        metric=metric,
        target_total_rules=target_total_rules,
        target_ratio=target_ratio,
        weight_by_spread=weight_by_spread,
    )
    scored_df = reformat_scored_df(scored_df, df)
    return scored_df


def get_trading_signals(
    scored_df: pd.DataFrame,
    initial_capital: float = 50000,
    threshold: float = 0.6,
    max_positions: int = 3,
    liquidity_threshold: float = 1000000,
    price_threshold: float = 2,
) -> pd.DataFrame:
    """
    Get trading signals for the current day based on scored data.

    Args:
        scored_df: DataFrame containing scored tickers with columns:
            - ticker
            - total_accuracy_score
            - orig_close
            - roll5_mean_intraday_total_dollar_volume_all
        initial_capital: Total capital to allocate
        threshold: Minimum score threshold for selection
        max_positions: Maximum number of positions to take
        liquidity_threshold: Minimum average daily volume in dollars
        price_threshold: Minimum price per share

    Returns:
        DataFrame with columns:
            - ticker: Stock symbol
            - score: Signal score
            - price: Entry price
            - quantity: Number of shares to buy
            - position_size: Dollar amount allocated
    """
    # Get latest date's data
    latest_date = scored_df["trade_date"].max()
    current_data = scored_df[scored_df["trade_date"] == latest_date].copy()

    # Apply filters
    qualified = current_data[
        (current_data["roll5_mean_intraday_total_dollar_volume_all"] >= liquidity_threshold)
        & (current_data["intraday_last_close_before_1555"] >= price_threshold)
        & (current_data["total_accuracy_score"] >= threshold)
    ]

    # Sort by score and volume
    qualified = qualified.sort_values(
        by=["total_accuracy_score", "roll5_mean_intraday_total_dollar_volume_all"], ascending=[False, False]
    )

    # Select top N positions
    selected = qualified.head(max_positions)

    if len(selected) == 0:
        return pd.DataFrame(columns=["ticker", "score", "price", "quantity", "position_size"])

    # Calculate position sizes (equal weight)
    position_size = initial_capital / len(selected)

    # Calculate quantities
    signals = pd.DataFrame(
        {
            "ticker": selected["ticker"],
            "score": selected["total_accuracy_score"],
            "price": selected["intraday_last_close_before_1555"],
            "position_size": position_size,
            "quantity": (position_size / selected["intraday_last_close_before_1555"]).astype(
                int
            ),  # Round down to whole shares
        }
    )

    # Ensure minimum quantity of 1
    signals["quantity"] = signals["quantity"].clip(lower=1)

    # Recalculate actual position sizes based on rounded quantities
    signals["position_size"] = signals["quantity"] * signals["price"]

    return signals.reset_index(drop=True)
