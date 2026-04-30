"""Financial backtesting and strategy evaluation utilities."""
import numpy as np
import pandas as pd


def calculate_strategy_metrics(returns_series, strategy_name="Strategy"):
    """Calculate comprehensive performance metrics for a returns series."""
    total_return = returns_series.sum()
    win_count = np.sum(returns_series > 0)
    loss_count = np.sum(returns_series < 0)
    trade_count = len(returns_series)
    win_rate = win_count / trade_count if trade_count > 0 else 0

    avg_win = returns_series[returns_series > 0].mean() if win_count > 0 else 0
    avg_loss = returns_series[returns_series < 0].mean() if loss_count > 0 else 0

    # Profit Factor
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
    test_df_copy['strat_long_short'] = 0
    test_df_copy.loc[test_df_copy[pred_col] == 1, 'strat_long_short'] = test_df_copy[price_col]
    test_df_copy.loc[test_df_copy[pred_col] == -1, 'strat_long_short'] = -test_df_copy[price_col]

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
