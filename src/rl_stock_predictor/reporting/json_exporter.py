"""JSON export for prediction results with full historical data."""
import json
from pathlib import Path
from typing import List
from datetime import datetime

from ..prediction.results import PredictionResult


def export_results_to_json(
    results: List[PredictionResult],
    output_path: Path = None,
    include_history: bool = True,
    indent: int = 2
) -> Path:
    """
    Export prediction results to JSON file.

    JSON format includes full historical data (net worth, actions, prices)
    which is useful for detailed analysis and visualization.

    Args:
        results: List of PredictionResult objects
        output_path: Path to save JSON (if None, auto-generate in outputs/reports/)
        include_history: Include historical data arrays (default: True)
        indent: JSON indentation level (default: 2)

    Returns:
        Path to saved JSON file

    Example:
        >>> results = predictor.predict_batch(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31')
        >>> json_path = export_results_to_json(results)
        >>> print(f"Saved to {json_path}")
    """
    if not results:
        raise ValueError("No results to export")

    # Auto-generate output path if not provided
    if output_path is None:
        from ..utils.paths import REPORTS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(results) == 1:
            filename = f"prediction_{results[0].ticker}_{timestamp}.json"
        else:
            filename = f"prediction_batch_{timestamp}.json"
        output_path = REPORTS_DIR / filename

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to dictionaries
    results_data = [r.to_dict(include_history=include_history) for r in results]

    # Create output structure with metadata
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_predictions": len(results),
            "tickers": [r.ticker for r in results],
            "include_history": include_history,
        },
        "predictions": results_data,
    }

    # Add summary statistics for batch predictions
    if len(results) > 1:
        output_data["summary"] = {
            "avg_profit_pct": sum(r.profit_pct for r in results) / len(results),
            "avg_sharpe_ratio": sum(r.sharpe_ratio for r in results) / len(results),
            "avg_max_drawdown": sum(r.max_drawdown for r in results) / len(results),
            "avg_win_rate": sum(r.win_rate for r in results) / len(results),
            "avg_model_vs_benchmark": sum(r.model_vs_benchmark for r in results) / len(results),
            "best_performer": max(results, key=lambda r: r.profit_pct).ticker,
            "worst_performer": min(results, key=lambda r: r.profit_pct).ticker,
        }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=indent)

    print(f"\nJSON report saved to: {output_path}")
    return output_path


def export_single_result_to_json(
    result: PredictionResult,
    output_path: Path = None,
    include_history: bool = True
) -> Path:
    """
    Export a single prediction result to JSON.

    Convenience function for single-stock predictions.

    Args:
        result: Single PredictionResult object
        output_path: Path to save JSON (if None, auto-generate)
        include_history: Include historical data arrays

    Returns:
        Path to saved JSON file
    """
    return export_results_to_json([result], output_path, include_history)
