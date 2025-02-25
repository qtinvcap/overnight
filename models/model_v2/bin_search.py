import numpy as np
import pandas as pd
from typing import Callable, Dict, Union

def get_performance_metric(metric_name: str) -> Callable:

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
    
    def trimmed_mean_std_ratio(x):
        trimmed = x[(x >= x.quantile(0.01)) & (x <= x.quantile(0.99))]
        mean = trimmed.mean()
        std = trimmed.std()
        return mean / std if std != 0 else 0
    
    metrics = {
        'mean': lambda x: x.mean(),
        'median': lambda x: x.median(),
        'mean_std_ratio': mean_std_ratio,
        'median_std_ratio': median_std_ratio,
        'mean_abs_std_ratio': mean_abs_std_ratio,
        'median_abs_std_ratio': median_abs_std_ratio,
        'trimmed_mean_std_ratio': trimmed_mean_std_ratio
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(metrics.keys())}")
    
    return metrics[metric_name]

def analyze_feature_range(
    df: pd.DataFrame,
    feature_name: str,
    performance_col: str = 'perf_close_to_open',
    metric: Union[str, Callable] = 'mean',
    n_ranges: int = 10,
    method: str = 'tree',
    min_samples_pct: float = 0.05  # 5% of data minimum per leaf
) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    from sklearn.tree import DecisionTreeRegressor
    
    if method not in ['quantile', 'linear', 'tree']:
        raise ValueError("Method must be one of: 'quantile', 'linear', 'tree'")


    categorical_features = ['ticker', 'day_of_week', 'month', 'year', 'is_short_above_long',
                          "hammer_flag", "doji_flag", "is_short_above_extra_long"]
    
    if isinstance(metric, str):
        perf_metric = get_performance_metric(metric)
        metric_name = metric
    else:
        perf_metric = metric
        metric_name = metric.__name__
    
    def calculate_accuracy(x):
        """Calculate percentage of positive or zero returns"""
        return (x >= 0).mean()
    
    is_categorical = (
        df[feature_name].dtype == 'object' or 
        df[feature_name].dtype.name == 'category' or 
        feature_name in categorical_features
    )
    
    if is_categorical:
        range_stats = df.groupby(feature_name).agg({
            performance_col: ['count', 'std', perf_metric, calculate_accuracy]
        }).round(6)
        range_stats.columns = ['count', 'std', metric_name, 'accuracy']
        range_stats['min_value'] = range_stats.index
        range_stats['max_value'] = range_stats.index
        return range_stats
    
    if not np.issubdtype(df[feature_name].dtype, np.number):
        raise ValueError(f"Feature {feature_name} is neither numeric nor categorical")
    
    feature_data = df[feature_name].replace([np.inf, -np.inf], np.nan)

    if method == 'tree':
        # Fit decision tree
        min_samples = int(len(df) * min_samples_pct)
        tree = DecisionTreeRegressor(
            #criterion='absolute_error',
            max_leaf_nodes=n_ranges,
            min_samples_leaf=min_samples  # At least 5% of data in each leaf
        )
        
        # Reshape for sklearn
        X = feature_data.values.reshape(-1, 1)
        y = df[performance_col].values
        y = (y - np.median(y)) / (np.quantile(y, 0.75) - np.quantile(y, 0.25))
        valid = (y < np.quantile(y, 0.99))
        
        # Fit tree and get split points
        tree.fit(X[valid], y[valid])
        
        # Extract split points from tree
        def get_tree_thresholds(tree):
            thresholds = []
            def recurse(node):
                if tree.children_left[node] != -1:  # Not a leaf
                    thresholds.append(tree.threshold[node])
                    recurse(tree.children_left[node])
                    recurse(tree.children_right[node])
            recurse(0)
            return sorted(set(thresholds))
        
        range_edges = [-np.inf] + get_tree_thresholds(tree.tree_) + [np.inf]
        range_edges = pd.Series(sorted(set(range_edges)))

    if (method == 'quantile') or (method == 'tree' and len(range_edges) < 3) :
        if method == 'tree':
            print(f"Tree method used for {feature_name} with {len(range_edges)} ranges")
        quantiles = np.linspace(0, 1, n_ranges + 1)
        range_edges = feature_data.quantile(quantiles)
        range_edges = pd.Series(sorted(range_edges.unique()))
    elif method == 'linear':
        min_val = feature_data.min()
        max_val = feature_data.max()
        if min_val == max_val:
            raise ValueError(f"Feature {feature_name} has no variance")
        range_edges = np.linspace(min_val, max_val, n_ranges + 1)
    
    if len(range_edges) < 3:
        raise ValueError(f"Not enough unique values in {feature_name}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['range'] = pd.cut(feature_data,
                            bins=range_edges,
                            include_lowest=True,
                            duplicates='drop')
    
    range_stats = df.groupby('range', observed=False).agg({
        performance_col: ['count', 'std', perf_metric, calculate_accuracy],
        feature_name: ['min', 'max']
    }).round(6)
    
    # Flatten column names
    range_stats.columns = ['count', 'std', metric_name, 'accuracy', 'min_value', 'max_value']
    
    return range_stats

def find_best_feature_ranges(
    df: pd.DataFrame,
    features_list: list,
    performance_col: str = 'perf_close_to_open',
    metric: Union[str, Callable] = 'mean',
    n_ranges: int = 10,
    min_samples: int = 1000,
    method: str = 'tree',
    verbose: bool = False
) -> Dict:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    best_ranges = {}
    worst_ranges = {}
    metric_name = metric if isinstance(metric, str) else metric.__name__
    
    for feature in features_list:
        try:
            # Skip features with too many NaN values
            nan_ratio = df[feature].isna().mean()
            if nan_ratio > 0.5:
                print(f"Skipping {feature}: too many NaN values ({nan_ratio:.2%})")
                continue
            
            # Get range statistics
            stats = analyze_feature_range(
                df=df,
                feature_name=feature,
                performance_col=performance_col,
                metric=metric,
                n_ranges=n_ranges,
                method=method
            )
            
            # Filter ranges with enough samples
            valid_ranges = stats[stats['count'] >= min_samples]
            
            if not valid_ranges.empty:
                for range_type in ['best', 'worst']:
                    # Find range with highest performance
                    range_idx = valid_ranges[metric_name].idxmax() if range_type == 'best' else valid_ranges[metric_name].idxmin()
                    optimal_metric = valid_ranges.loc[range_idx, metric_name]
                    sample_count = valid_ranges.loc[range_idx, 'count']
                    std_value = valid_ranges.loc[range_idx, 'std']
                    accuracy = valid_ranges.loc[range_idx, 'accuracy']
                    
                    # Get range bounds
                    range_min = valid_ranges.loc[range_idx, 'min_value']
                    range_max = valid_ranges.loc[range_idx, 'max_value']
                    
                    value = {
                            'range': range_idx,
                            f'{metric_name}_performance': optimal_metric,
                            'std': std_value,
                            'accuracy': accuracy,
                            'sample_count': sample_count,
                            'min_value': range_min,
                            'max_value': range_max
                        }
                    if range_type == 'best':
                        best_ranges[feature] = value
                    else:
                        worst_ranges[feature] = value
            else:
                print(f"Skipping {feature}: no ranges with enough samples")
                
        except Exception as e:
            print(f"Error analyzing {feature}: {str(e)}")
            continue
    
    # Sort dictionary by performance
    sorted_best_ranges = dict(sorted(
        best_ranges.items(),
        key=lambda x: x[1][f'{metric_name}_performance'],
        reverse=True
    ))
    sorted_worst_ranges = dict(sorted(
        worst_ranges.items(),
        key=lambda x: x[1][f'{metric_name}_performance'],
        reverse=False
    ))
    
    # Print results
    if sorted_best_ranges and verbose:
        print(f"\n=== Best Ranges by Feature (Sorted by |{metric_name}|) ===")
        print("\n{:<30} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Feature", f"{metric_name}", "Std", "Accuracy", "Samples", "Min Value", "Max Value"
        ))
        print("-" * 115)
        
        for feature, stats in sorted_best_ranges.items():
            print("{:<30} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.0f} {:<15} {:<15}".format(
                feature[:30],
                stats[f'{metric_name}_performance'],
                stats['std'],
                stats['accuracy'],
                stats['sample_count'],
                str(stats['min_value'])[:14],
                str(stats['max_value'])[:14]
            ))
    elif verbose:
        print("\nNo valid ranges found for any feature")
    
    return sorted_best_ranges, sorted_worst_ranges

import numpy as np
import pandas as pd
from typing import Callable, Dict, Union

def create_scoring_features(
    df: pd.DataFrame, 
    optimal_ranges: Union[dict, pd.DataFrame], 
    rule_type: str = 'best',
    metric: str = 'mean_std_ratio_performance', 
    limit_performance: float = 0.06,
    top_n_perf: int = None,
    verbose: bool = False
) -> tuple[pd.DataFrame, list, dict]:
    """
    Create accuracy-based scoring features based on good and bad performing ranges.
    
    Args:
        df: Input DataFrame with features
        best_ranges: Dictionary or DataFrame containing range information for features
        metric: Name of the performance metric column
        limit_performance: Minimum performance threshold for good ranges
        max_negative_performance: Maximum performance threshold for bad ranges
        
    Returns:
        tuple: (DataFrame with new features, list of binary feature names, rules information)
    """
    assert rule_type in ['best', 'worst']

    # Setup and validation
    if not isinstance(optimal_ranges, pd.DataFrame):
        optimal_ranges = pd.DataFrame.from_dict(optimal_ranges, orient='index')
    
    if metric not in optimal_ranges.columns:
        raise ValueError(f"Metric '{metric}' not found in best_ranges columns: {optimal_ranges.columns.tolist()}")
    
    required_cols = ['range', 'min_value', 'max_value', 'accuracy']
    missing_cols = [col for col in required_cols if col not in optimal_ranges.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in best_ranges: {missing_cols}")
    
    # Initialize tracking variables
    accuracy_features = []
    rules_info = {
        f'rules_{rule_type}': [],
    }
    
    # Define categorical features
    categorical_features = [
        'ticker', 'day_of_week', 'month', 'year', 
        'is_short_above_long', 'is_short_above_extra_long',
        'hammer_flag', 'doji_flag'
    ]
    
    def process_categorical_feature(feature: str, row: pd.Series) -> tuple[str, pd.Series]:
        """Helper function to process categorical features"""
        category = row['range']
        accuracy = row['accuracy']
        
        feature_name = f"{feature}_in_{rule_type}_range"
        binary_values = (df[feature] == category).astype(float) * accuracy
        condition = f"{feature} == {category}"
            
        return feature_name, binary_values, condition
    
    def process_numerical_feature(feature: str, row: pd.Series) -> tuple[str, pd.Series]:
        """Helper function to process numerical features"""
        min_val = row['min_value'].left if hasattr(row['min_value'], 'left') else row['min_value']
        max_val = row['max_value'].right if hasattr(row['max_value'], 'right') else row['max_value']
        accuracy = row['accuracy']
        
        feature_name = f"{feature}_in_{rule_type}_range"
        binary_values = ((df[feature] >= min_val) & (df[feature] <= max_val)).astype(float) #* accuracy
        condition = f"{min_val:.4f} <= {feature} <= {max_val:.4f}"

            
        return feature_name, binary_values, condition
    
    def process_features(features_df: pd.DataFrame):
        """Process a set of features and create accuracy-weighted indicators"""
        
        for feature, row in features_df.iterrows():
            try:
                is_categorical = (
                    feature in categorical_features or 
                    df[feature].dtype.name in ['object', 'category']
                )
                
                if is_categorical:
                    feature_name, accuracy_values, condition = process_categorical_feature(
                        feature, row
                    )
                    feature_type = 'categorical'
                else:
                    feature_name, accuracy_values, condition = process_numerical_feature(
                        feature, row
                    )
                    feature_type = 'numerical'
                
                # Add accuracy-weighted feature to DataFrame
                df[feature_name] = accuracy_values
                accuracy_features.append(feature_name)
                
                # Record rule information with appropriate accuracy
                used_accuracy = row['accuracy']
                rules_info[f'rules_{rule_type}'].append({
                    'feature': feature,
                    'type': feature_type,
                    'condition': condition,
                    'performance': row[metric],
                    'original_accuracy': row['accuracy'],
                    'used_accuracy': used_accuracy
                })
                
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue
    
    # Process good and bad performing ranges
    if top_n_perf is not None:
        limit_performance = sorted(optimal_ranges[metric], reverse=(rule_type == 'best'))[top_n_perf]

    optimal_ranges_df = optimal_ranges[
        (optimal_ranges[metric] >= limit_performance) if rule_type == 'best' else (optimal_ranges[metric] <= limit_performance)
    ]
    
    process_features(optimal_ranges_df)
    
    # Calculate total accuracy score
    df['total_accuracy_score'] = df[accuracy_features].sum(axis=1)
    df['accuracy_score_pct'] = (
        df['total_accuracy_score'] / len(accuracy_features) if accuracy_features else 0.0
    )
    
    # Print summary
    if verbose:
        print_rules_summary(rules_info, accuracy_features)
    
    return df, accuracy_features, rules_info


def print_rules_summary(rules_info: dict, accuracy_features: list):
    """Print a formatted summary of the rules"""
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
        rules_df = rules_df.sort_values('used_accuracy', ascending=False)
        
        for _, rule in rules_df.iterrows():
            print(f"{rule['feature']} ({rule['type']})")
            print(f"Condition: {rule['condition']}")
            print(f"Performance: {rule['performance']:.4f}")
            print(f"Original Accuracy: {rule['original_accuracy']:.4f}")
            print(f"Used Accuracy: {rule['used_accuracy']:.4f}\n")
    
    print_rules_section(rules_info['good_rules'], "Good Rules (Sorted by Accuracy)")
    print_rules_section(rules_info['bad_rules'], "Bad Rules to Avoid (Sorted by Accuracy)")


import numpy as np

def add_momentum_ranking(df,liquidity_threshold, price_threshold):
    """
    Calculate 5-day close-to-close performance and rank tickers by this performance for each day.
    Rank 1 = highest performance.
    """

    df = df[df['roll5_mean_intraday_total_dollar_volume_all'] > liquidity_threshold]
    df = df[df['orig_close'] > price_threshold]

    # Calculate 5-day close-to-close performance
    df = df.sort_values(['ticker', 'trade_date']).copy()
    df['close_5d_ago'] = df.groupby('ticker')['close'].shift(5)
    df['perf_5d'] = df['close'] / df['close_5d_ago'] - 1
    
    # Rank tickers by performance for each day (1 = highest performance)
    df['momentum_ranking'] = df.groupby('trade_date')['perf_5d'].rank(ascending=False)
    
    # Clean up intermediate columns
    df = df.drop(['close_5d_ago', 'perf_5d'], axis=1)
    
    return df

def backtest_with_commission(
    df,
    threshold=0.35,
    max_positions=10,
    initial_capital=50000,
    date_col='trade_date',
    open_col='orig_close',       # buy price at day's close
    next_open_col='next_day_open',  # sell price next day
    probas_col='total_range_score',
    liquidity_threshold=1000000,
    price_threshold=3,
    top_n_momentum=30,
    score_similarity_threshold=0.1  # New parameter for monitoring
):
    if top_n_momentum is not None:
        df = add_momentum_ranking(df, liquidity_threshold, price_threshold)
        df = df[df["momentum_ranking"] < top_n_momentum]
    # Sort by date
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    grouped = df_sorted.groupby(date_col, as_index=False)
    
    # Initialize tracking variables
    daily_records = []
    positions_list = []  # For tracking individual positions
    similar_scores_events = []  # For monitoring similar scores
    capital = initial_capital
    unique_dates = df_sorted[date_col].unique()

    def net_return_with_commission(raw_return):
        """Calculate net return after commission"""
        entry_after_commission = 1 + 0.0015  # Pay 1.5% more on entry
        exit_after_commission = 1 - 0.0015
        return (1 + raw_return) * exit_after_commission / entry_after_commission - 1

    def monitor_similar_scores(qualified_df, selected_positions, day, similarity_threshold):
        """Monitor and log occurrences of similar scores"""
        if len(qualified_df) <= len(selected_positions):
            return
            
        cutoff_score = selected_positions[probas_col].min()
        max_score = selected_positions[probas_col].max()
        
        # Find positions with similar scores to those selected
        similar_scores_mask = (
            (qualified_df[probas_col] >= cutoff_score - similarity_threshold) &
            (qualified_df[probas_col] <= max_score + similarity_threshold)
        )
        similar_scores_df = qualified_df[similar_scores_mask]
        
        if len(similar_scores_df) > len(selected_positions):
            similar_scores_events.append({
                'date': day,
                'n_similar_positions': len(similar_scores_df),
                'n_selected': len(selected_positions),
                'score_range': f"{similar_scores_df[probas_col].min():.4f} to {similar_scores_df[probas_col].max():.4f}",
                'selected_scores': selected_positions[probas_col].tolist(),
                'all_similar_scores': similar_scores_df[probas_col].tolist(),
                'selected_tickers': selected_positions['ticker'].tolist(),
                'all_similar_tickers': similar_scores_df['ticker'].tolist()
            })

    # Loop through each trading day
    for day in unique_dates:
        day_data = grouped.get_group(day)
        
        # Filter for minimum liquidity
        qualified_liquidity = day_data[
            day_data['roll5_mean_intraday_total_dollar_volume_all'] > liquidity_threshold
        ]
        qualified_price = qualified_liquidity[
            qualified_liquidity['orig_close'] > price_threshold
        ]

        qualified_price = qualified_price.copy()  # Create copy to avoid SettingWithCopyWarning
        qualified_price[probas_col] = qualified_price[probas_col]#.round(1)

          # Sort by both criteria first
        qualified_price = qualified_price.sort_values(
            by=[probas_col, 'roll5_mean_intraday_total_dollar_volume_all'],
            ascending=[False, False]
        )

   
        # Filter by probability threshold
        qualified = qualified_price[
            qualified_price[probas_col] >= threshold
        ].dropna(subset=[open_col, next_open_col, probas_col])

        
        if len(qualified) == 0:
            daily_return = 0.0
            n_positions = 0
        else:
            
            topN = qualified.head(max_positions)

            # Select top N positions
            #topN = qualified.sort_values(probas_col, #ascending=False).head(max_positions)
            n_positions = len(topN)

            position_size = initial_capital / n_positions
            
            # Monitor similar scores
            #monitor_similar_scores(qualified, topN, day, score_similarity_threshold)
            
            # Calculate returns
            raw_returns = topN[next_open_col] / topN[open_col] - 1.0
            net_returns = raw_returns.apply(net_return_with_commission)
            daily_return = net_returns.mean()
            
            # Record individual positions
            for idx, row in topN.iterrows():
                raw_return = row[next_open_col] / row[open_col] - 1.0
                net_return = net_return_with_commission(raw_return)
                
                position_info = {
                    'date': day,
                    'ticker': row['ticker'],
                    'entry_price': row[open_col],
                    'exit_price': row[next_open_col],
                    'raw_return': raw_return,
                    'net_return': net_return,
                    'probability': row[probas_col],
                    'volume': row['roll5_mean_intraday_total_dollar_volume_all'],
                    'position_size': position_size,
                    'pnl': position_size * net_return
                }
                positions_list.append(position_info)

        # Update capital
        capital = capital * (1.0 + daily_return)
        
        # Record daily summary
        daily_records.append({
            'date': day,
            'daily_return': daily_return,
            'capital': capital,
            'n_positions': n_positions
        })

    # Create DataFrames
    daily_df = pd.DataFrame(daily_records)
    daily_df.sort_values('date', inplace=True)
    daily_df.set_index('date', inplace=True)

    positions_df = pd.DataFrame(positions_list)
    if len(positions_df) > 0:
        positions_df['win'] = positions_df['net_return'] > 0
    
    # Calculate performance metrics
    daily_df['cum_return'] = daily_df['capital'] / initial_capital - 1.0
    
    # Drawdown calculation
    running_max = daily_df['capital'].cummax()
    daily_df['drawdown'] = 1 - (daily_df['capital'] / running_max)
    max_dd = daily_df['drawdown'].max()

    # Annualized metrics
    total_days = len(daily_df)
    if total_days < 2:
        annual_return = np.nan
    else:
        final_cum_return = daily_df['cum_return'].iloc[-1]
        annual_return = (1 + final_cum_return)**(252 / total_days) - 1

    daily_std = daily_df['daily_return'].std()
    annual_vol = daily_std * np.sqrt(252)

    mar_ratio = annual_return / max_dd

    # Sharpe Ratio
    if not np.isnan(annual_vol) and annual_vol > 1e-12:
        sharpe = annual_return / annual_vol
    else:
        sharpe = np.nan

    # Compile statistics
    stats = {
        'final_capital': capital,
        'final_cum_return': daily_df['cum_return'].iloc[-1],
        'max_drawdown': max_dd,
        'annualized_return': annual_return,
        'mar_ratio': mar_ratio,
        'annualized_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'total_trades': len(positions_df),
        'win_rate': positions_df['win'].mean() if len(positions_df) > 0 else 0,
        'avg_return_per_trade': positions_df['net_return'].mean() if len(positions_df) > 0 else 0,
        'avg_positions_per_day': daily_df['n_positions'].mean(),
        'sortino_ratio': calculate_sortino(daily_df['daily_return']),
    }

    # Create similar scores monitoring DataFrame
    similar_scores_df = pd.DataFrame(similar_scores_events)
    
    # Print summary of similar scores events
    if len(similar_scores_events) > 0:
        print("\n=== Similar Scores Monitoring ===")
        print(f"Total trading days: {len(unique_dates)}")
        print(f"Days with similar scores: {len(similar_scores_events)}")
        print(f"Percentage of days with similar scores: {(len(similar_scores_events)/len(unique_dates))*100:.2f}%")
        
        print("\nSample of similar scores events:")
        pd.set_option('display.max_columns', None)
        print(similar_scores_df.head())
        
        # Additional statistics
        avg_similar = similar_scores_df['n_similar_positions'].mean()
        print(f"\nAverage number of similar positions when event occurs: {avg_similar:.2f}")
    else:
        print("\nNo instances of similar scores found")

    return daily_df, stats, positions_df, similar_scores_df

def analyze_annual_performance(daily_results):
    """
    Analyze and plot annual performance metrics from backtest results
    """
    # Add year to daily_df if not already present
    daily_results = daily_results.copy()
    daily_results['year'] = daily_results.index.year
    
    # Calculate annual performance
    annual_performance = daily_results.groupby('year').apply(
        lambda x: {
            'return': (1 + x['daily_return']).prod() - 1,
            'sharpe': x['daily_return'].mean() / x['daily_return'].std() * np.sqrt(252) if len(x) > 1 else 0,
            'max_drawdown': x['drawdown'].max(),
            'avg_positions': x['n_positions'].mean()
        }
    ).apply(pd.Series)
    
    # Plot annual performance
    plt.figure(figsize=(12, 6))
    annual_performance['return'].plot(kind='bar')
    plt.title('Annual Returns')
    plt.xlabel('Year')
    plt.ylabel('Return')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add return values on top of bars
    for i, v in enumerate(annual_performance['return']):
        plt.text(i, v, f'{v:.1%}', ha='center', va='bottom' if v > 0 else 'top')
    
    plt.tight_layout()
    plt.show()
    
    return annual_performance

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
            "predicted_score",
            "trade_date",
            "orig_close",
            "next_day_open",
            "roll5_mean_intraday_close_auction_dollar_volume",
            "roll5_mean_intraday_total_dollar_volume_all",
            "intraday_last_close_before_1555",
        ]
    ]

def calculate_sortino(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio from daily returns.
    
    Args:
        daily_returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0)
        
    Returns:
        float: Sortino ratio
    """
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = daily_returns - daily_rf
    
    # Calculate average daily excess return
    avg_excess_return = excess_returns.mean()
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt((negative_returns ** 2).mean())
    
    # Handle case where there are no negative returns
    if downside_std == 0:
        return np.inf if avg_excess_return > 0 else -np.inf
    
    # Calculate annualized Sortino ratio
    sortino = avg_excess_return / downside_std * np.sqrt(252)
    
    return sortino
