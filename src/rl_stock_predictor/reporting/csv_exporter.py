"""CSV export for prediction results."""
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime

from ..prediction.results import PredictionResult


def export_results_to_csv(
    results: List[PredictionResult],
    output_path: Path = None,
    include_history: bool = False
) -> Path:
    """
    Export prediction results to CSV file.

    Args:
        results: List of PredictionResult objects
        output_path: Path to save CSV (if None, auto-generate in outputs/reports/)
        include_history: Include historical data columns (makes file much larger)

    Returns:
        Path to saved CSV file

    Example:
        >>> results = predictor.predict_batch(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31')
        >>> csv_path = export_results_to_csv(results)
        >>> print(f"Saved to {csv_path}")
    """
    if not results:
        raise ValueError("No results to export")

    # Auto-generate output path if not provided
    if output_path is None:
        from ..utils.paths import REPORTS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(results) == 1:
            filename = f"prediction_{results[0].ticker}_{timestamp}.csv"
        else:
            filename = f"prediction_batch_{timestamp}.csv"
        output_path = REPORTS_DIR / filename

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame([r.to_series() for r in results])

    # Reorder columns for readability
    column_order = [
        # Identification
        'ticker',
        'start_date',
        'end_date',

        # Performance
        'initial_balance',
        'final_net_worth',
        'profit_amount',
        'profit_pct',

        # Risk metrics
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'max_drawdown',
        'volatility',

        # Trading stats
        'total_trades',
        'num_buys',
        'num_sells',
        'win_rate',
        'profit_factor',
        'avg_holding_period',

        # Benchmark comparison
        'buy_hold_final_value',
        'buy_hold_profit_pct',
        'model_vs_benchmark',

        # RL metrics
        'total_steps',
        'cumulative_reward',
    ]

    # Reorder columns (keep any extra columns at the end)
    available_cols = [col for col in column_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in column_order]
    df = df[available_cols + extra_cols]

    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"\nCSV report saved to: {output_path}")
    return output_path


def export_single_result_to_csv(
    result: PredictionResult,
    output_path: Path = None
) -> Path:
    """
    Export a single prediction result to CSV.

    Convenience function for single-stock predictions.

    Args:
        result: Single PredictionResult object
        output_path: Path to save CSV (if None, auto-generate)

    Returns:
        Path to saved CSV file
    """
    return export_results_to_csv([result], output_path)
