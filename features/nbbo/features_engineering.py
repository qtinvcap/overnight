import numpy as np


def compute_nbbo_features(quotes):
    """
    Given a list of NBBO quote dictionaries from Polygon,
    compute additional aggregates that might be useful for your strategy.
    Returns a dict of feature metrics.
    """

    if not quotes:
        # Return something consistent (all NaNs or zeros) if no quotes
        return {
            "avg_spread": np.nan,
            "max_spread": np.nan,
            "spread_volatility": np.nan,
            "mid_price_volatility": np.nan,
            "avg_dollar_liquidity_ask": np.nan,
            "median_dollar_liquidity_ask": np.nan,
            "min_dollar_liquidity_ask": np.nan,
            "max_dollar_liquidity_ask": np.nan,
            "avg_dollar_liquidity_bid": np.nan,
            "imbalance": np.nan,
        }

    ask_prices = []
    bid_prices = []
    ask_sizes_shares = []
    bid_sizes_shares = []
    dollar_at_asks = []
    dollar_at_bids = []
    spreads = []
    mid_prices = []

    for q in quotes:
        ask_price = q.get("ask_price", 0.0)
        bid_price = q.get("bid_price", 0.0)
        ask_size_rl = q.get("ask_size", 0)
        bid_size_rl = q.get("bid_size", 0)

        # Convert round lot to actual shares
        ask_size_sh = ask_size_rl * 100
        bid_size_sh = bid_size_rl * 100

        spread = ask_price - bid_price
        mid_price = (ask_price + bid_price) / 2.0 if (ask_price > 0 and bid_price > 0) else 0.0
        dollar_at_ask = ask_price * ask_size_sh
        dollar_at_bid = bid_price * bid_size_sh

        ask_prices.append(ask_price)
        bid_prices.append(bid_price)
        ask_sizes_shares.append(ask_size_sh)
        bid_sizes_shares.append(bid_size_sh)
        dollar_at_asks.append(dollar_at_ask)
        dollar_at_bids.append(dollar_at_bid)
        spreads.append(spread)
        mid_prices.append(mid_price)

    # Convert to numpy for quick stats
    spreads = np.array(spreads)
    mid_prices = np.array(mid_prices)
    dollar_at_asks = np.array(dollar_at_asks)
    dollar_at_bids = np.array(dollar_at_bids)

    # Basic aggregates
    avg_spread = np.mean(spreads) if len(spreads) > 0 else np.nan
    max_spread = np.max(spreads) if len(spreads) > 0 else np.nan
    spread_vol = np.std(spreads) if len(spreads) > 0 else np.nan
    mid_price_vol = np.std(mid_prices) if len(mid_prices) > 0 else np.nan

    avg_dollar_ask = np.mean(dollar_at_asks) if len(dollar_at_asks) > 0 else np.nan
    median_dollar_ask = np.median(dollar_at_asks) if len(dollar_at_asks) > 0 else np.nan
    min_dollar_ask = np.min(dollar_at_asks) if len(dollar_at_asks) > 0 else np.nan
    max_dollar_ask = np.max(dollar_at_asks) if len(dollar_at_asks) > 0 else np.nan
    avg_dollar_bid = np.mean(dollar_at_bids) if len(dollar_at_bids) > 0 else np.nan

    sum_dollar_ask = np.sum(dollar_at_asks)
    sum_dollar_bid = np.sum(dollar_at_bids)
    if (sum_dollar_ask + sum_dollar_bid) != 0:
        imbalance = (sum_dollar_bid - sum_dollar_ask) / (sum_dollar_bid + sum_dollar_ask)
    else:
        imbalance = np.nan

    features = {
        "avg_spread": avg_spread,
        "max_spread": max_spread,
        "spread_volatility": spread_vol,
        "mid_price_volatility": mid_price_vol,
        "avg_dollar_liquidity_ask": avg_dollar_ask,
        "median_dollar_liquidity_ask": median_dollar_ask,
        "min_dollar_liquidity_ask": min_dollar_ask,
        "max_dollar_liquidity_ask": max_dollar_ask,
        "avg_dollar_liquidity_bid": avg_dollar_bid,
        "imbalance": imbalance,
    }
    return features
