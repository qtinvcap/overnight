import pandas as pd
from models.model_v3.bin_search import create_scoring_features, reformat_scored_df
import pickle
from sklearn.linear_model import LinearRegression
import os
import numpy as np

def load_best_ranges(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def score_features_df(
    df: pd.DataFrame,
    train_ranges_lst: list[dict],
    opt_features_lst: list[list],
    linReg_lst: list[LinearRegression],
    range_names: list[int],
    top_n_perf: int = 100,
) -> pd.DataFrame:
    out_predicted_scores = {}
    for range_name, train_ranges, opt_features, linReg in zip(range_names, train_ranges_lst, opt_features_lst, linReg_lst):
        df, _, _ = create_scoring_features(
            df=df,
            optimal_ranges=train_ranges['best'], 
            rule_type='best',
            metric='trimmed_mean_std_ratio_performance', 
            top_n_perf=top_n_perf
        )

        df, _, _ = create_scoring_features(
            df=df,
            optimal_ranges=train_ranges['worst'], 
            rule_type='worst',
            metric='trimmed_mean_std_ratio_performance', 
            top_n_perf=top_n_perf
        )

        features_best = opt_features[f'features_best']
        features_worst = opt_features[f'features_worst']

        df["intraday_last5_price_acceleration_in_best_range"] = 0
        df["intraday_last5_price_acceleration_in_worst_range"] = 0

        X_best = df[features_best].values
        X_worst = df[features_worst].values
        X = np.concatenate([X_best, -X_worst], axis=1)

        out_predicted_scores[f"{range_name}"] = linReg.predict(X)    
        
    df['predicted_score'] = (
        (
            (out_predicted_scores['upper_n_lower_quantile0.02_w_etf_noPrice'] > 0.05) *  (out_predicted_scores['upper_quantile0.005_w_etf_noPrice'] > 0.0)
        ) * out_predicted_scores['upper_n_lower_quantile0.02_w_etf_noPrice']
        + (out_predicted_scores['upper_quantile0.005_v2'] > 0.5) * out_predicted_scores['upper_quantile0.005_v2']
    )
    
    scored_df = reformat_scored_df(df)
    return scored_df


def get_trading_signals(
    scored_df: pd.DataFrame,
    initial_capital: float = 50000,
    threshold: float = 1e-6,
    max_positions: int = 5,
    liquidity_threshold: float = 2000000,
    price_threshold: float = 1,
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
        & (current_data["predicted_score"] >= threshold)
        & (current_data["intraday_total_dollar_volume_until_1555"] > liquidity_threshold)
    ]

    # Sort by score and volume
    qualified = qualified.sort_values(
        by=["predicted_score", "roll5_mean_intraday_total_dollar_volume_all"], ascending=[False, False]
    )

    # Select top N positions
    selected = qualified.head(max_positions)

    if len(selected) == 0:
        return pd.DataFrame(columns=["ticker", "score", "price", "quantity", "position_size"])
    
    # Calculate position sizes (equal weight)
    position_size = initial_capital / max_positions

    # Calculate quantities
    signals = pd.DataFrame(
        {
            "ticker": selected["ticker"],
            "score": selected["predicted_score"],
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


def main_inference(
    df: pd.DataFrame,
    initial_capital: float = 50000,
    threshold: float = 1e-6,
    max_positions: int = 5,
    liquidity_threshold: float = 10_000_000,
    price_threshold: float = 3,
) -> pd.DataFrame:
    # Load the best ranges
    range_names = [
        "upper_quantile0.005_v2",
        "upper_quantile0.005_w_etf_noPrice",
        "upper_n_lower_quantile0.02_w_etf_noPrice"
    ]
    optimal_ranges_paths = [os.path.join(
        os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")), f"models/model_v3/rules/rules_and_weights_lr_2015_2024_quantile_50_100_{p}.pkl"
    ) for p in range_names]

    train_ranges_lst, opt_features_lst, linReg_lst = zip(*[load_best_ranges(path) for path in optimal_ranges_paths])

    # Score data    
    scored_df = score_features_df(
        df=df,
        train_ranges_lst=train_ranges_lst,
        opt_features_lst=opt_features_lst,
        linReg_lst=linReg_lst,
        range_names=range_names,
    )

    return get_trading_signals(
        scored_df=scored_df,
        initial_capital=initial_capital,
        threshold=threshold,
        max_positions=max_positions,
        liquidity_threshold=liquidity_threshold,
        price_threshold=price_threshold,
    )