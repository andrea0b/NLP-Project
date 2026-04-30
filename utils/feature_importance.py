"""Feature importance and backtest visualization utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def extract_top_features(model, vectorizer, n_features=15):
    """Extract top positive and negative features for each sentiment class.

    Args:
        model: Trained classifier with coef_ attribute
        vectorizer: Fitted TF-IDF vectorizer with get_feature_names_out()
        n_features: Number of top features to extract per class

    Returns:
        Dict mapping class labels to {'positive': [...], 'negative': [...]} feature lists
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_

    class_names = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    sentiment_features = {}

    for class_idx, class_label in enumerate([-1, 0, 1]):
        class_coefs = coefficients[class_idx]

        top_positive_indices = np.argsort(class_coefs)[-n_features:][::-1]
        top_positive_words = [(feature_names[i], class_coefs[i]) for i in top_positive_indices]

        top_negative_indices = np.argsort(class_coefs)[:n_features]
        top_negative_words = [(feature_names[i], class_coefs[i]) for i in top_negative_indices]

        sentiment_features[class_label] = {
            'positive': top_positive_words,
            'negative': top_negative_words
        }

    return sentiment_features, class_names


def print_top_features(sentiment_features, class_names):
    """Print top features for each sentiment class in a readable format."""
    print("\n" + "="*100)
    print("TOP INFLUENTIAL WORDS BY SENTIMENT CLASS")
    print("="*100)

    for class_label in [-1, 0, 1]:
        features = sentiment_features[class_label]
        print(f"\n{'─'*100}")
        print(f"CLASS: {class_names[class_label].upper()}")
        print(f"{'─'*100}")

        print(f"\n  ✓ WORDS STRONGLY ASSOCIATED WITH {class_names[class_label].upper()} (High Coefficients):")
        for i, (word, coef) in enumerate(features['positive'], 1):
            print(f"    {i:2d}. {word:25s} coef: {coef:+.4f}")

        print(f"\n  ✗ WORDS AGAINST {class_names[class_label].upper()} (Low Coefficients):")
        for i, (word, coef) in enumerate(features['negative'], 1):
            print(f"    {i:2d}. {word:25s} coef: {coef:+.4f}")

    print("\n" + "="*100 + "\n")


def plot_features_by_class(model, vectorizer, n_features=12):
    """Create side-by-side bar charts of top features for each sentiment class."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Top 12 Most Important Words per Sentiment Class', fontsize=14, fontweight='bold', y=1.02)

    class_list = [-1, 0, 1]
    class_names = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    colors = ['#d62728', '#808080', '#2ca02c']

    for ax_idx, class_label in enumerate(class_list):
        ax = axes[ax_idx]
        class_coefs = coefficients[ax_idx]

        top_indices = np.argsort(np.abs(class_coefs))[-n_features:][::-1]
        top_words = feature_names[top_indices]
        top_coefs = class_coefs[top_indices]

        bar_colors = [colors[ax_idx] if coef > 0 else '#cccccc' for coef in top_coefs]

        y_pos = np.arange(len(top_words))
        ax.barh(y_pos, top_coefs, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words, fontsize=10)
        ax.set_xlabel('Model Coefficient', fontsize=10)
        ax.set_title(f'{class_names[class_label].upper()} Sentiment', fontsize=12, fontweight='bold', pad=10)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        for i, (word, coef) in enumerate(zip(top_words, top_coefs)):
            ax.text(coef, i, f'  {coef:.3f}', va='center', fontsize=8,
                    ha='left' if coef > 0 else 'right')

    plt.tight_layout()
    plt.show()

    print("Chart Legend:")
    print("  • Darker colors: Words strongly associated with the sentiment")
    print("  • Lighter colors: Words against the sentiment (appear in opposite class)")
    print("  • Coefficient magnitude indicates word importance\n")


def plot_feature_heatmap(model, vectorizer, n_features=20):
    """Create a heatmap showing top features across all sentiment classes."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_

    fig, ax = plt.subplots(figsize=(14, 10))

    max_abs_coefs = np.max(np.abs(coefficients), axis=0)
    top_indices = np.argsort(max_abs_coefs)[-n_features:][::-1]

    heatmap_data = coefficients[:, top_indices].T
    heatmap_words = feature_names[top_indices]

    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'], fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(len(heatmap_words)))
    ax.set_yticklabels(heatmap_words, fontsize=10)
    ax.set_ylabel('Words (TF-IDF Features)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sentiment Class', fontsize=11, fontweight='bold')
    ax.set_title('Feature Coefficient Heatmap: Top 20 Most Important Words', fontsize=12, fontweight='bold', pad=15)

    for i in range(len(heatmap_words)):
        for j in range(3):
            ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Model Coefficient', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("Heatmap Interpretation:")
    print("  • Blue regions: Negative coefficient (word evidence AGAINST that class)")
    print("  • Red regions: Positive coefficient (word evidence FOR that class)")
    print("  • Darker shades: Stronger influence on classification")
    print("  • Notice how words cluster around their dominant sentiment class\n")


def plot_backtest_results(test_df_full, cum_perf):
    """Create comprehensive backtest visualization with 5 subplots."""
    print("Generating comprehensive backtest visualization...\n")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_perf.index, cum_perf['strat_long_short'], label='Long-Short (SVM)',
             color='#2ca02c', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    ax1.plot(cum_perf.index, cum_perf['strat_long_only'], label='Long-Only (SVM)',
             color='#1f77b4', linewidth=2.5, linestyle='--', marker='s', markersize=3, alpha=0.8)
    ax1.plot(cum_perf.index, cum_perf['benchmark'], label='Benchmark (Buy & Hold)',
             color='#d62728', linewidth=2, linestyle=':', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_title('Strategy Cumulative Returns Over Test Period', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # 2. Return distribution by sentiment
    ax2 = fig.add_subplot(gs[1, 0])
    returns_data = [
        test_df_full[test_df_full['pred_svm'] == 1]['realized_return'] * 100,
        test_df_full[test_df_full['pred_svm'] == -1]['realized_return'] * 100,
        test_df_full[test_df_full['pred_svm'] == 0]['realized_return'] * 100
    ]
    bp = ax2.boxplot(returns_data, labels=['Positive\nSignal', 'Negative\nSignal', 'Neutral\nSignal'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#2ca02c', '#d62728', '#808080']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Daily Return (%)', fontsize=10)
    ax2.set_title('Return Distribution by Sentiment', fontsize=11, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Win rate
    ax3 = fig.add_subplot(gs[1, 1])
    win_rates = {
        'Long-Short': np.sum(test_df_full['strat_long_short'] > 0) / len(test_df_full),
        'Long-Only': np.sum(test_df_full['strat_long_only'] > 0) / len(test_df_full),
        'Benchmark': np.sum(test_df_full['benchmark'] > 0) / len(test_df_full)
    }
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    bars = ax3.bar(win_rates.keys(), win_rates.values(), color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Win Rate', fontsize=10)
    ax3.set_title('Win Rate Comparison', fontsize=11, fontweight='bold')
    ax3.set_ylim([0, 0.7])
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Signal distribution
    ax4 = fig.add_subplot(gs[2, 0])
    signal_counts = {
        'Long': np.sum(test_df_full['pred_svm'] == 1),
        'Short': np.sum(test_df_full['pred_svm'] == -1),
        'Neutral': np.sum(test_df_full['pred_svm'] == 0)
    }
    colors_sig = ['#2ca02c', '#d62728', '#808080']
    bars = ax4.bar(signal_counts.keys(), signal_counts.values(), color=colors_sig, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Signal Distribution in Test Period', fontsize=11, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Profit by signal
    ax5 = fig.add_subplot(gs[2, 1])
    profit_data = {
        'Long\nSignals': test_df_full[test_df_full['pred_svm'] == 1]['realized_return'].sum() * 100,
        'Short\nSignals': -test_df_full[test_df_full['pred_svm'] == -1]['realized_return'].sum() * 100,
        'Neutral\nHolds': test_df_full[test_df_full['pred_svm'] == 0]['realized_return'].sum() * 100
    }
    colors_profit = ['#2ca02c', '#d62728', '#808080']
    bars = ax5.bar(profit_data.keys(), profit_data.values(), color=colors_profit, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Total Return (%)', fontsize=10)
    ax5.set_title('Return by Signal Type', fontsize=11, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=10, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Trading Strategy Backtest Results: Sentiment-Driven Signals', fontsize=14, fontweight='bold', y=0.995)
    plt.show()

    print("✓ Visualization complete\n")
