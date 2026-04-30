"""Financial backtesting and strategy evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calculate_strategy_metrics(returns_series, strategy_name="Strategy"):
    """Calculate comprehensive performance metrics for a returns series."""
    win_count = np.sum(returns_series > 0)
    loss_count = np.sum(returns_series < 0)
    trade_count = len(returns_series)
    win_rate = win_count / trade_count if trade_count > 0 else 0

    avg_win = returns_series[returns_series > 0].mean() if win_count > 0 else 0
    avg_loss = returns_series[returns_series < 0].mean() if loss_count > 0 else 0

    # Total return using compounding (correct method)
    total_return = (1 + returns_series).prod() - 1

    # Profit Factor (using sum for wins/losses)
    profit_sum = returns_series[returns_series > 0].sum()
    loss_sum = abs(returns_series[returns_series < 0].sum())
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else 0

    # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
    std_return = returns_series.std()
    sharpe = (returns_series.mean() * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'Strategy': strategy_name,
        'Total Return (%)': total_return * 100,
        'Num Trades': trade_count,
        'Win Rate (%)': win_rate * 100,
        'Avg Win (%)': avg_win * 100,
        'Avg Loss (%)': avg_loss * 100,
        'Profit Factor': profit_factor,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown * 100
    }


def backtest_strategies(test_df, price_col='realized_return', pred_col='pred_svm'):
    """Create strategy returns from sentiment predictions and realized returns."""
    test_df_copy = test_df.copy()

    # Strategy 1: Long Only on Positive Predictions
    test_df_copy['strat_long_only'] = np.where(
        test_df_copy[pred_col] == 1,
        test_df_copy[price_col],
        0
    )

    # Strategy 2: Long-Short (Long on Pos, Short on Neg)
    test_df_copy['strat_long_short'] = 0.0
    test_df_copy.loc[test_df_copy[pred_col] == 1, 'strat_long_short'] = test_df_copy.loc[test_df_copy[pred_col] == 1, price_col].values
    test_df_copy.loc[test_df_copy[pred_col] == -1, 'strat_long_short'] = -test_df_copy.loc[test_df_copy[pred_col] == -1, price_col].values

    # Benchmark: Always Long (Buy and Hold)
    test_df_copy['benchmark'] = test_df_copy[price_col]

    return test_df_copy


def compute_cumulative_performance(test_df, date_col='date'):
    """Compute cumulative returns over time by date."""
    daily_perf = test_df.groupby(date_col)[['strat_long_only', 'strat_long_short', 'benchmark']].mean()
    cum_perf = (1 + daily_perf).cumprod() - 1
    return cum_perf


def sentiment_breakdown(test_df, pred_col='pred_svm', return_col='realized_return'):
    """Analyze performance by sentiment class."""
    results = []
    for sentiment_val, sentiment_name in [(-1, 'Negative'), (0, 'Neutral'), (1, 'Positive')]:
        mask = test_df[pred_col] == sentiment_val
        subset = test_df[mask]

        if len(subset) > 0:
            results.append({
                'Sentiment': sentiment_name,
                'Count': len(subset),
                'Avg Return (%)': subset[return_col].mean() * 100,
                'Total Return (%)': subset[return_col].sum() * 100,
                'Win Rate (%)': (np.sum(subset[return_col] > 0) / len(subset)) * 100,
                'Std Dev (%)': subset[return_col].std() * 100
            })

    return pd.DataFrame(results)


def extract_confidence_scores(decision_values):
    """Extract normalized confidence scores (0-1) from model decision function.

    Args:
        decision_values: Array of decision function values from classifier.
                        Can be 1D (binary) or 2D (multiclass).

    Returns:
        Array of normalized confidence scores (0-1) with shape (n_samples,)
    """
    decision_values = np.asarray(decision_values)

    # For multiclass, take max absolute value across classes per sample
    if decision_values.ndim > 1:
        abs_values = np.max(np.abs(decision_values), axis=1)  # (n_samples,)
    else:
        abs_values = np.abs(decision_values)  # (n_samples,)

    # Normalize to 0-1 using min-max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    confidence = scaler.fit_transform(abs_values.reshape(-1, 1)).flatten()

    return confidence


def backtest_strategies_confidence_weighted(test_df, price_col='realized_return',
                                            pred_col='pred_svm', confidence_col='confidence',
                                            target_leverage=1.5, date_col='date'):
    """Build trading strategies using confidence-weighted position sizing.

    Position sizing:
    - Each day, non-neutral signals sized by confidence
    - Total gross leverage targets ~target_leverage (e.g., 1.5 = 150% deployed)
    - Scales naturally: more signals → smaller per-signal, fewer signals → larger per-signal

    Strategies:
    - Long-Only: Takes long positions on positive signals, cash on others
    - Long-Short: Long on positive, short on negative, cash on neutral
    - Benchmark: Equal-weight buy-hold of all unique stocks in test period

    Args:
        test_df: DataFrame with columns [date, ticker, pred_col, price_col, confidence_col, realized_return]
        price_col: Column name for daily returns
        pred_col: Column name for predictions (-1, 0, 1)
        confidence_col: Column name for confidence scores (0-1)
        target_leverage: Target gross leverage (1.0 = 100%, 1.5 = 150%)
        date_col: Column name for dates

    Returns:
        DataFrame with added columns: strat_long_only, strat_long_short, benchmark, position_size
    """
    test_df_copy = test_df.copy()
    test_df_copy[date_col] = pd.to_datetime(test_df_copy[date_col])

    # Initialize position_size column
    test_df_copy['position_size'] = 0.0

    # Calculate position sizes for each day
    for day_date in test_df_copy[date_col].unique():
        day_mask = test_df_copy[date_col] == day_date
        day_df = test_df_copy[day_mask]

        non_neutral = day_df[day_df[pred_col] != 0]

        if len(non_neutral) > 0:
            longs = non_neutral[non_neutral[pred_col] == 1]
            shorts = non_neutral[non_neutral[pred_col] == -1]

            long_strength = longs[confidence_col].sum()
            short_strength = shorts[confidence_col].sum()
            total_strength = long_strength + short_strength

            if total_strength > 0:
                scale = target_leverage / total_strength

                # Assign position sizes
                test_df_copy.loc[longs.index, 'position_size'] = longs[confidence_col].values * scale
                test_df_copy.loc[shorts.index, 'position_size'] = -shorts[confidence_col].values * scale

    # Calculate strategy returns using position sizes
    test_df_copy['strat_long_short'] = test_df_copy['position_size'] * test_df_copy[price_col]
    test_df_copy['strat_long_only'] = np.where(
        test_df_copy['position_size'] > 0,  # only on long positions
        test_df_copy['position_size'] * test_df_copy[price_col],
        0
    )

    # Benchmark: Equal-weight buy-hold of ALL unique stocks in test period
    # Each stock gets 1/n_stocks weight, return = weight * stock_return
    n_stocks = test_df_copy['ticker'].nunique()
    test_df_copy['benchmark'] = (1.0 / n_stocks) * test_df_copy[price_col]

    return test_df_copy


def compute_cumulative_performance_by_date(test_df, date_col='date'):
    """Compute cumulative returns over time by date (for daily rebalancing strategies).

    Aggregates daily returns (average across all trades/signals), then compounds.
    """
    test_df_copy = test_df.copy()
    # Ensure date column is properly formatted
    test_df_copy[date_col] = pd.to_datetime(test_df_copy[date_col])

    # Group by date and calculate mean returns
    daily_perf = test_df_copy.groupby(test_df_copy[date_col].dt.date)[
        ['strat_long_only', 'strat_long_short', 'benchmark']
    ].mean()

    # Compound returns over time
    cum_perf = (1 + daily_perf).cumprod() - 1
    return cum_perf
