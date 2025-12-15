"""Console output formatting for prediction results."""
import pandas as pd
from typing import List

from ..prediction.results import PredictionResult


def format_single_result(result: PredictionResult, detailed: bool = True) -> str:
    """
    Format a single prediction result for console output.

    Args:
        result: PredictionResult object
        detailed: Include all metrics (default: True)

    Returns:
        Formatted string for console display

    Example:
        >>> result = predictor.predict_single('AAPL', '2024-01-01', '2024-12-31')
        >>> print(format_single_result(result))
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"PREDICTION RESULTS: {result.ticker}")
    lines.append("=" * 70)

    # Period
    lines.append(f"\nPeriod: {result.start_date} to {result.end_date}")
    lines.append(f"Trading days: {result.total_steps}")

    # Performance
    lines.append("\nPERFORMANCE:")
    lines.append(f"  Initial balance:     ${result.initial_balance:>12,.2f}")
    lines.append(f"  Final net worth:     ${result.final_net_worth:>12,.2f}")
    lines.append(f"  Profit/Loss:         ${result.profit_amount:>12,.2f} ({result.profit_pct:+.2f}%)")

    if detailed:
        # Risk metrics
        lines.append("\nRISK METRICS:")
        lines.append(f"  Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
        lines.append(f"  Sortino Ratio:       {result.sortino_ratio:>12.2f}")
        lines.append(f"  Calmar Ratio:        {result.calmar_ratio:>12.2f}")
        lines.append(f"  Max Drawdown:        {result.max_drawdown * 100:>12.1f}%")
        lines.append(f"  Volatility:          {result.volatility * 100:>12.1f}% (annualized)")

        # Trading statistics
        lines.append("\nTRADING STATISTICS:")
        lines.append(f"  Total trades:        {result.total_trades:>12}")
        lines.append(f"  Buy actions:         {result.num_buys:>12}")
        lines.append(f"  Sell actions:        {result.num_sells:>12}")
        lines.append(f"  Win rate:            {result.win_rate * 100:>12.1f}%")
        lines.append(f"  Profit factor:       {result.profit_factor:>12.2f}")
        lines.append(f"  Avg holding period:  {result.avg_holding_period:>12.1f} days")

    # Benchmark comparison
    lines.append("\nBENCHMARK COMPARISON:")
    lines.append(f"  Buy-and-hold value:  ${result.buy_hold_final_value:>12,.2f}")
    lines.append(f"  Buy-and-hold return: {result.buy_hold_profit_pct:>12.2f}%")
    lines.append(f"  Model return:        {result.profit_pct:>12.2f}%")

    # Highlight outperformance
    if result.model_vs_benchmark > 0:
        lines.append(f"  Outperformance:      {result.model_vs_benchmark:>12.2f}% ✓")
    else:
        lines.append(f"  Underperformance:    {result.model_vs_benchmark:>12.2f}% ✗")

    if detailed:
        # RL metrics
        lines.append("\nRL METRICS:")
        lines.append(f"  Cumulative reward:   {result.cumulative_reward:>12.2f}")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_batch_summary(results: List[PredictionResult], top_n: int = None) -> str:
    """
    Format batch prediction results for console output.

    Shows summary table with key metrics and highlights best/worst performers.

    Args:
        results: List of PredictionResult objects
        top_n: Only show top N performers (default: show all)

    Returns:
        Formatted string for console display

    Example:
        >>> results = predictor.predict_batch(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-12-31')
        >>> print(format_batch_summary(results))
    """
    if not results:
        return "No results to display"

    lines = []

    # Header
    lines.append("\n" + "=" * 100)
    lines.append(f"BATCH PREDICTION SUMMARY ({len(results)} stocks)")
    lines.append("=" * 100)

    # Create DataFrame for easy formatting
    df = pd.DataFrame([r.to_series() for r in results])

    # Sort by profit percentage
    df = df.sort_values('profit_pct', ascending=False)

    if top_n:
        df = df.head(top_n)

    # Select columns for display
    display_cols = [
        'ticker',
        'profit_pct',
        'buy_hold_profit_pct',
        'model_vs_benchmark',
        'sharpe_ratio',
        'max_drawdown',
        'total_trades',
        'win_rate',
    ]

    # Format columns
    df_display = df[display_cols].copy()
    df_display['profit_pct'] = df_display['profit_pct'].apply(lambda x: f"{x:+.2f}%")
    df_display['buy_hold_profit_pct'] = df_display['buy_hold_profit_pct'].apply(lambda x: f"{x:+.2f}%")
    df_display['model_vs_benchmark'] = df_display['model_vs_benchmark'].apply(lambda x: f"{x:+.2f}%")
    df_display['sharpe_ratio'] = df_display['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    df_display['max_drawdown'] = df_display['max_drawdown'].apply(lambda x: f"{x*100:.1f}%")
    df_display['win_rate'] = df_display['win_rate'].apply(lambda x: f"{x*100:.1f}%")

    # Rename columns for display
    df_display.columns = [
        'Ticker',
        'Model Return',
        'Buy-Hold Return',
        'Outperform',
        'Sharpe',
        'Max DD',
        'Trades',
        'Win Rate',
    ]

    lines.append("\n" + df_display.to_string(index=False))

    # Summary statistics
    lines.append("\n" + "-" * 100)
    lines.append("SUMMARY STATISTICS:")
    lines.append("-" * 100)

    avg_model_return = df['profit_pct'].mean()
    avg_benchmark_return = df['buy_hold_profit_pct'].mean()
    avg_outperformance = df['model_vs_benchmark'].mean()
    avg_sharpe = df['sharpe_ratio'].mean()
    avg_win_rate = df['win_rate'].mean()

    lines.append(f"Average model return:       {avg_model_return:>8.2f}%")
    lines.append(f"Average buy-hold return:    {avg_benchmark_return:>8.2f}%")
    lines.append(f"Average outperformance:     {avg_outperformance:>8.2f}%")
    lines.append(f"Average Sharpe ratio:       {avg_sharpe:>8.2f}")
    lines.append(f"Average win rate:           {avg_win_rate*100:>8.1f}%")

    # Best/worst performers
    best = results[df['profit_pct'].idxmax()]
    worst = results[df['profit_pct'].idxmin()]

    lines.append(f"\nBest performer:   {best.ticker:>6} ({best.profit_pct:+.2f}%)")
    lines.append(f"Worst performer:  {worst.ticker:>6} ({worst.profit_pct:+.2f}%)")

    # Count stocks that beat benchmark
    beat_benchmark = (df['model_vs_benchmark'] > 0).sum()
    total_stocks = len(df)

    lines.append(f"\nStocks beating benchmark: {beat_benchmark}/{total_stocks} ({beat_benchmark/total_stocks*100:.1f}%)")

    lines.append("=" * 100)

    return "\n".join(lines)


def format_comparison_table(results: List[PredictionResult]) -> str:
    """
    Format a simple comparison table of model vs benchmark returns.

    Args:
        results: List of PredictionResult objects

    Returns:
        Formatted comparison table string
    """
    if not results:
        return "No results to compare"

    lines = []
    lines.append("\nMODEL VS BENCHMARK COMPARISON:")
    lines.append("-" * 60)
    lines.append(f"{'Ticker':<10} {'Model Return':<15} {'B&H Return':<15} {'Diff':<10}")
    lines.append("-" * 60)

    for result in results:
        lines.append(
            f"{result.ticker:<10} "
            f"{result.profit_pct:>6.2f}% {'':>6} "
            f"{result.buy_hold_profit_pct:>6.2f}% {'':>6} "
            f"{result.model_vs_benchmark:>+6.2f}%"
        )

    lines.append("-" * 60)

    return "\n".join(lines)
