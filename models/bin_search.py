import numpy as np
import pandas as pd
from typing import Union, Callable
from joblib import Parallel, delayed


def get_performance_metric(metric_name: str) -> Callable:
    """
    Returns a function f(x: pd.Series) -> float that calculates the desired metric.
    """

    def mean_std_ratio(x):
        mean = x.mean()
        std = x.std()
        return mean / std if std != 0 else 0

    def median_std_ratio(x):
        median = x.median()
        std = x.std()
        return median / std if std != 0 else 0

    def mean_abs_std_ratio(x):
        mean = x.mean()
        std = x.std()
        return abs(mean) / std if std != 0 else 0

    def median_abs_std_ratio(x):
        median = x.median()
        std = x.std()
        return abs(median) / std if std != 0 else 0

    # We'll skip "trimmed_mean_std_ratio" here because we'll implement it
    # as a custom aggregator rather than a simple function applied columnwise.
    # But we leave it in this dictionary just for consistency:
    def trimmed_mean_std_ratio(x):
        # If you need to call it directly for some reason:
        trimmed = x[(x >= x.quantile(0.01)) & (x <= x.quantile(0.99))]
        mean = trimmed.mean()
        std = trimmed.std()
        return mean / std if std != 0 else 0

    metrics = {
        "mean": lambda x: x.mean(),
        "median": lambda x: x.median(),
        "mean_std_ratio": mean_std_ratio,
        "median_std_ratio": median_std_ratio,
        "mean_abs_std_ratio": mean_abs_std_ratio,
        "median_abs_std_ratio": median_abs_std_ratio,
        "trimmed_mean_std_ratio": trimmed_mean_std_ratio,
    }

    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}")

    return metrics[metric_name]


def analyze_one_feature(
    df: pd.DataFrame,
    feature_name: str,
    performance_col: str = "perf_close_to_open",
    metric: Union[str, Callable] = "mean",
    n_ranges: int = 10,
    method: str = "quantile",
    min_samples: int = 1000,
    categorical_features=None,
) -> pd.DataFrame:
    """
    Analyzes a single feature by binning (or grouping if categorical),
    computing count, std, metric, accuracy, and min/max within each bin.
    Returns a DataFrame of stats for each bin (or category).
    """
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if categorical_features is None:
        categorical_features = [
            "ticker",
            "day_of_week",
            "month",
            "year",
            "is_short_above_long",
            "hammer_flag",
            "doji_flag",
            "is_short_above_extra_long",
        ]

    # Convert metric to callable if needed
    if isinstance(metric, str):
        metric_func = get_performance_metric(metric)
        metric_name = metric
    else:
        metric_func = metric
        metric_name = metric.__name__

    def calculate_accuracy(x):
        """Calculate percentage of x >= 0."""
        return (x >= 0).mean()

    # Check if feature is categorical
    is_categorical = (
        (df[feature_name].dtype == "object")
        or (df[feature_name].dtype.name == "category")
        or (feature_name in categorical_features)
    )

    # For performance reasons, we can restrict df to only relevant columns
    # to reduce memory overhead during groupby:
    df_local = df[[feature_name, performance_col]].copy()

    # Drop inf values and remain consistent
    df_local[feature_name] = df_local[feature_name].replace([np.inf, -np.inf], np.nan)
    df_local = df_local.dropna(subset=[feature_name, performance_col])

    # Quick check for too few non-NaN rows
    if df_local.shape[0] < min_samples:
        # Return an empty DataFrame as a sign to skip
        return pd.DataFrame()

    # === If categorical: simple groupby
    if is_categorical:
        grouped = df_local.groupby(feature_name)
        stats_cat = (
            grouped[performance_col]
            .agg(count="count", std="std", metric_val=metric_func, accuracy=calculate_accuracy)
            .reset_index()
        )

        # For categorical, min and max of the feature are the same as the category label
        stats_cat["min_value"] = stats_cat[feature_name]
        stats_cat["max_value"] = stats_cat[feature_name]

        # Filter by min_samples
        stats_cat = stats_cat[stats_cat["count"] >= min_samples].set_index(feature_name)
        stats_cat.rename(columns={"metric_val": metric_name}, inplace=True)
        return stats_cat

    # === Otherwise numeric: define bins
    feature_data = df_local[feature_name]
    if method == "quantile":
        quantiles = np.linspace(0, 1, n_ranges + 1)
        range_edges = feature_data.quantile(quantiles)
        range_edges = pd.Series(sorted(range_edges.unique()))
    else:  # linear spacing
        min_val = feature_data.min()
        max_val = feature_data.max()
        if min_val == max_val:
            return pd.DataFrame()  # No variance, skip
        range_edges = np.linspace(min_val, max_val, n_ranges + 1)

    # Bin the data
    df_local["range_bin"] = pd.cut(feature_data, bins=range_edges, include_lowest=True, duplicates="drop")

    # Now define an aggregator function that handles either normal or trimmed metrics
    def aggregator(subdf: pd.DataFrame) -> pd.Series:
        perf = subdf[performance_col]

        # Count, std, accuracy apply to the entire bin
        c = len(perf)
        if c == 0:
            return pd.Series(dtype="float")

        s = perf.std()
        acc = (perf >= 0).mean()

        # Now compute the metric:
        if metric_name == "trimmed_mean_std_ratio":
            # Special trimming
            q1 = perf.quantile(0.01)
            q99 = perf.quantile(0.99)
            trimmed = perf[(perf >= q1) & (perf <= q99)]
            m = trimmed.mean()
            st = trimmed.std()
            metric_val = m / st if (st != 0) else 0

            # Also want min/max of the feature for the *trimmed subset*
            trimmed_feat = subdf.loc[trimmed.index, feature_name]
            if len(trimmed_feat) == 0:
                f_min = np.nan
                f_max = np.nan
            else:
                f_min = trimmed_feat.min()
                f_max = trimmed_feat.max()
        else:
            # For normal metrics, just call the metric function on the entire bin
            metric_val = metric_func(perf)
            f_min = subdf[feature_name].min()
            f_max = subdf[feature_name].max()

        return pd.Series(
            {"count": c, "std": s, metric_name: metric_val, "accuracy": acc, "min_value": f_min, "max_value": f_max}
        )

    # Compute the stats with one pass
    range_stats = (
        df_local.groupby("range_bin", observed=False)
        .apply(aggregator, include_groups=False)
        .dropna(how="all")  # In case some bins are empty
    )

    # Filter out small bins
    range_stats = range_stats[range_stats["count"] >= min_samples]

    return range_stats


def find_best_feature_ranges(
    df: pd.DataFrame,
    features_list: list,
    performance_col: str = "perf_close_to_open",
    metric: Union[str, Callable] = "mean",
    n_ranges: int = 10,
    min_samples: int = 1000,
    method: str = "quantile",
):
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    metric_name = metric if isinstance(metric, str) else metric.__name__
    feature_ranges = {}

    # We'll define a small helper function that runs analyze_one_feature
    def process_feature(feature):
        # Subset of df with only the columns we need to reduce pickling overhead
        df_subset = df[[feature, performance_col]].copy()

        # Skip if too many NaNs
        nan_ratio = df_subset[feature].isna().mean()
        if nan_ratio > 0.5:
            print(f"[SKIP] {feature} => too many NaNs: {nan_ratio:.1%}")
            return (feature, None)  # Return None so we know to skip

        # Run the aggregator
        stats = analyze_one_feature(
            df_subset,
            feature_name=feature,
            performance_col=performance_col,
            metric=metric,
            n_ranges=n_ranges,
            method=method,
            min_samples=min_samples,
        )
        # If empty or not enough data, skip
        if stats.empty:
            return (feature, None)
        else:
            return (feature, stats)

    # Run in parallel
    # n_jobs=64 tries to use all CPU cores (tweak as needed)
    results = Parallel(n_jobs=64)(delayed(process_feature)(f) for f in features_list)

    # Build up the feature_ranges dictionary from results
    for feature, stats in results:
        if stats is None or stats.empty:
            # Means we skip
            continue

        # Identify best/worst bin
        best_idx = stats[metric_name].idxmax()
        worst_idx = stats[metric_name].idxmin()

        best_metric = stats.loc[best_idx, metric_name]
        best_std = stats.loc[best_idx, "std"]
        best_acc = stats.loc[best_idx, "accuracy"]
        best_count = stats.loc[best_idx, "count"]
        best_min = stats.loc[best_idx, "min_value"]
        best_max = stats.loc[best_idx, "max_value"]

        worst_metric = stats.loc[worst_idx, metric_name]
        worst_std = stats.loc[worst_idx, "std"]
        worst_acc = stats.loc[worst_idx, "accuracy"]
        worst_count = stats.loc[worst_idx, "count"]
        worst_min = stats.loc[worst_idx, "min_value"]
        worst_max = stats.loc[worst_idx, "max_value"]

        feature_ranges[feature] = {
            "best_range": {
                "range": best_idx,
                f"{metric_name}_performance": best_metric,
                "std": best_std,
                "accuracy": best_acc,
                "sample_count": best_count,
                "min_value": best_min,
                "max_value": best_max,
            },
            "worst_range": {
                "range": worst_idx,
                f"{metric_name}_performance": worst_metric,
                "std": worst_std,
                "accuracy": worst_acc,
                "sample_count": worst_count,
                "min_value": worst_min,
                "max_value": worst_max,
            },
            "performance_spread": best_metric - worst_metric,
        }

    # Sort dictionary by performance spread
    sorted_ranges = dict(sorted(feature_ranges.items(), key=lambda x: abs(x[1]["performance_spread"]), reverse=True))

    # Print results
    if sorted_ranges:
        print(f"\n=== Feature Ranges (Sorted by Performance Spread) ===")
        header_fmt = "{:<30} {:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}"
        row_fmt = "{:<30} {:<20.6f} {:<15.6f} {:<15.6f} {:<15.0f} {:<15} {:<15} {:<15.6f}"
        row2_fmt = "{:<30} {:<20.6f} {:<15.6f} {:<15.6f} {:<15.0f} {:<15} {:<15}"

        print(
            header_fmt.format(
                "Feature", f"{metric_name}", "Std", "Accuracy", "Samples", "Min Value", "Max Value", "Spread"
            )
        )
        print("-" * 140)

        for feature, stats in sorted_ranges.items():
            br = stats["best_range"]
            wr = stats["worst_range"]
            spread = stats["performance_spread"]
            print(
                row_fmt.format(
                    feature[:30] + " (BEST)",
                    br[f"{metric_name}_performance"],
                    br["std"],
                    br["accuracy"],
                    br["sample_count"],
                    str(br["min_value"])[:14],
                    str(br["max_value"])[:14],
                    spread,
                )
            )
            print(
                row2_fmt.format(
                    feature[:30] + " (WORST)",
                    wr[f"{metric_name}_performance"],
                    wr["std"],
                    wr["accuracy"],
                    wr["sample_count"],
                    str(wr["min_value"])[:14],
                    str(wr["max_value"])[:14],
                )
            )
            print("-" * 140)
    else:
        print("\nNo valid ranges found for any feature.")

    return sorted_ranges


def print_rules_summary(rules_info: dict, accuracy_features: list):
    """Print a formatted summary of the rules with their weights and performance metrics."""
    print("\n=== Rules Summary ===")
    print(f"Total Rules: {len(accuracy_features)}")
    print(f"Good Rules: {len(rules_info['good_rules'])}")
    print(f"Bad Rules: {len(rules_info['bad_rules'])}")

    def print_rules_section(rules: list, section_title: str):
        print(f"\n=== {section_title} ===")
        if not rules:
            print("No rules found")
            return

        rules_df = pd.DataFrame(rules)
        # Sort by weight * performance for most impactful rules first
        rules_df["impact"] = rules_df["weight"] * rules_df["performance"].abs()
        rules_df = rules_df.sort_values("impact", ascending=False)

        for _, rule in rules_df.iterrows():
            print(f"\n{rule['feature']} ({rule['type']})")
            print(f"Condition: {rule['condition']}")
            print(f"Performance: {rule['performance']:.4f}")
            print(f"Weight: {rule['weight']:.4f}")
            print(f"Accuracy: {rule['accuracy']:.4f}")
            print(f"Impact Score: {rule['impact']:.4f}")

    print_rules_section(rules_info["good_rules"], "Good Rules (Sorted by Impact)")
    print_rules_section(rules_info["bad_rules"], "Bad Rules to Avoid (Sorted by Impact)")


def create_scoring_features(
    df: pd.DataFrame,
    best_ranges: dict,
    metric: str = "trimmed_mean_std_ratio_performance",
    target_total_rules: int = 160,
    target_ratio: float = 0.5,
    weight_by_spread: bool = True,
) -> tuple[pd.DataFrame, list, dict]:
    """
    Create accuracy-based scoring features with adaptive thresholds to control rule count.

    Args:
        target_total_rules: Desired total number of rules (good + bad)
        target_ratio: Target ratio of good rules to total rules (0.5 means equal split)
    """
    # Get all performance values sorted
    good_performances = sorted([info["best_range"][metric] for info in best_ranges.values()], reverse=True)

    bad_performances = sorted([info["worst_range"][metric] for info in best_ranges.values()])

    # Calculate target numbers
    target_good_rules = int(target_total_rules * target_ratio)
    target_bad_rules = target_total_rules - target_good_rules

    # Set adaptive thresholds based on percentiles
    if len(good_performances) > target_good_rules:
        min_performance = good_performances[target_good_rules - 1]
    else:
        min_performance = good_performances[-1] if good_performances else 0

    if len(bad_performances) > target_bad_rules:
        max_negative_performance = bad_performances[target_bad_rules - 1]
    else:
        max_negative_performance = bad_performances[-1] if bad_performances else 0

    print(f"Adaptive thresholds - Good: {min_performance:.4f}, Bad: {max_negative_performance:.4f}")

    # Initialize tracking variables
    accuracy_features = []
    rules_info = {"good_rules": [], "bad_rules": []}

    # Only keep required columns initially
    required_cols = [
        "ticker",
        "day_of_week",
        "month",
        "year",
        "is_short_above_long",
        "is_short_above_extra_long",
        "hammer_flag",
        "doji_flag",
    ]
    feature_cols = list(best_ranges.keys())
    df_subset = df[list(set(required_cols + feature_cols))].copy()

    # Define categorical features
    categorical_features = [
        "ticker",
        "day_of_week",
        "month",
        "year",
        "is_short_above_long",
        "is_short_above_extra_long",
        "hammer_flag",
        "doji_flag",
    ]

    # Initialize total_accuracy_score as numpy array
    total_score = np.zeros(len(df_subset))
    total_weights = 0

    def process_feature(feature: str, range_info: dict, is_good_rule: bool) -> tuple[float, str, float]:
        """Process feature and return weight, condition, and binary values"""
        is_categorical = feature in categorical_features or df_subset[feature].dtype.name in ["object", "category"]

        # Get performance spread as weight
        perf_spread = abs(best_ranges[feature]["performance_spread"])
        weight = perf_spread if weight_by_spread else 1.0

        min_val = range_info["min_value"]
        max_val = range_info["max_value"]

        if is_categorical:
            category = range_info["range"]
            if is_good_rule:
                binary_values = (df_subset[feature] == category).astype(float)
                condition = f"{feature} == {category}"
            else:
                binary_values = (df_subset[feature] != category).astype(float)
                condition = f"{feature} != {category}"
        else:
            if is_good_rule:
                binary_values = ((df_subset[feature] >= min_val) & (df_subset[feature] <= max_val)).astype(float)
                condition = f"{min_val:.4f} <= {feature} <= {max_val:.4f}"
            else:
                binary_values = ((df_subset[feature] < min_val) | (df_subset[feature] > max_val)).astype(float)
                condition = f"{feature} < {min_val:.4f} OR {feature} > {max_val:.4f}"

        return binary_values, condition, weight

    # Process features
    for feature, feature_info in best_ranges.items():
        try:
            # Process good rules
            if feature_info["best_range"][metric] >= min_performance:
                binary_values, condition, weight = process_feature(
                    feature, feature_info["best_range"], is_good_rule=True
                )

                # Update total score directly
                total_score += binary_values * weight
                total_weights += weight

                accuracy_features.append(f"{feature}_in_good_range")
                rules_info["good_rules"].append(
                    {
                        "feature": feature,
                        "type": "categorical" if feature in categorical_features else "numerical",
                        "condition": condition,
                        "performance": feature_info["best_range"][metric],
                        "weight": weight,
                        "accuracy": feature_info["best_range"]["accuracy"],
                    }
                )

            # Process bad rules
            if feature_info["worst_range"][metric] <= max_negative_performance:
                binary_values, condition, weight = process_feature(
                    feature, feature_info["worst_range"], is_good_rule=False
                )

                # Update total score directly
                total_score += binary_values * weight
                total_weights += weight

                accuracy_features.append(f"{feature}_outside_bad_range")
                rules_info["bad_rules"].append(
                    {
                        "feature": feature,
                        "type": "categorical" if feature in categorical_features else "numerical",
                        "condition": condition,
                        "performance": feature_info["worst_range"][metric],
                        "weight": weight,
                        "accuracy": feature_info["worst_range"]["accuracy"],
                    }
                )

        except Exception as e:
            print(f"Error processing feature {feature}: {str(e)}")
            continue

    # Create final DataFrame with only necessary columns
    df_new = pd.DataFrame(
        {
            "total_accuracy_score": total_score / total_weights if total_weights > 0 else total_score,
            "accuracy_score_pct": pd.Series(total_score).rank(pct=True),
        },
        index=df_subset.index,
    )

    # Print summary
    print_rules_summary(rules_info, accuracy_features)

    return df_new, accuracy_features, rules_info


def reformat_scored_df(scored_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Reformat the scored dataframe to include required columns"""
    scored_df["ticker"] = df["ticker"]
    scored_df["trade_date"] = df["trade_date"]
    scored_df["orig_close"] = df["orig_close"]
    scored_df["close"] = df["close"]
    scored_df["next_day_open"] = df["next_day_open"]
    scored_df["roll5_mean_intraday_close_auction_dollar_volume"] = df["roll5_mean_intraday_close_auction_dollar_volume"]
    scored_df["roll5_mean_intraday_total_dollar_volume_all"] = df["roll5_mean_intraday_total_dollar_volume_all"]
    scored_df["intraday_last_close_before_1555"] = df["intraday_last_close_before_1555"]
    return scored_df[
        [
            "ticker",
            "total_accuracy_score",
            "trade_date",
            "orig_close",
            "next_day_open",
            "roll5_mean_intraday_close_auction_dollar_volume",
            "roll5_mean_intraday_total_dollar_volume_all",
            "intraday_last_close_before_1555",
        ]
    ]


# Example usage:
# scored_df, binary_features, rules_info = create_scoring_features(
#    df=df,
#    best_ranges=best_ranges,
#    metric='trimmed_mean_std_ratio_performance',
#    #min_performance=0.06,
#    #max_negative_performance=0,
#    target_total_rules=160,
#    target_ratio=0.5,
#    weight_by_spread=False
# )
