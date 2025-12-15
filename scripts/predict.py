"""Enhanced prediction script with batch mode, metrics, and benchmarking."""
import argparse
from pathlib import Path
from dotenv import load_dotenv

from rl_stock_predictor import BatchPredictor
from rl_stock_predictor.reporting import (
    export_results_to_csv,
    export_results_to_json,
    format_single_result,
    format_batch_summary,
)
from rl_stock_predictor.utils.paths import MODELS_DIR

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run predictions with trained RL stock trading model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stock prediction
  python scripts/predict.py --ticker AAPL --start 2024-01-01 --end 2024-12-31

  # Batch prediction on multiple stocks
  python scripts/predict.py --tickers AAPL MSFT GOOGL --start 2024-01-01 --end 2024-12-31

  # Save results to CSV and JSON
  python scripts/predict.py --ticker AAPL --start 2024-01-01 --end 2024-12-31 --save-csv --save-json

  # Quiet mode (no progress output)
  python scripts/predict.py --ticker AAPL --start 2024-01-01 --end 2024-12-31 --quiet
        """
    )

    # Ticker selection (mutually exclusive)
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--ticker",
        type=str,
        help="Single stock ticker to predict on (e.g., AAPL)"
    )
    ticker_group.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Multiple stock tickers for batch prediction (e.g., AAPL MSFT GOOGL)"
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date for prediction data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date for prediction data (YYYY-MM-DD)"
    )

    # Model selection
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_multiasset_trader",
        help="Name of the model to load (from models/ directory, without .zip extension)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Full path to model file (overrides --model-name)"
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference (default: auto-detect)"
    )

    # Output options
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV file in outputs/reports/"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results to JSON file in outputs/reports/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for reports (default: outputs/reports/)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output during prediction"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=True,
        help="Show detailed results (default: True)"
    )

    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()

    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = MODELS_DIR / f"{args.model_name}.zip"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"Available models in {MODELS_DIR}:")
        for model_file in MODELS_DIR.glob("*.zip"):
            print(f"  - {model_file.stem}")
        return 1

    # Initialize BatchPredictor
    print("=" * 70)
    print("RL STOCK PREDICTOR - PREDICTION MODE")
    print("=" * 70)
    print(f"Model: {model_path.name}")
    print(f"Period: {args.start} to {args.end}")

    predictor = BatchPredictor(
        model_path=str(model_path),
        device=args.device
    )

    # Determine mode: single or batch
    if args.ticker:
        # Single stock mode
        print(f"Mode: Single stock ({args.ticker})")
        print("=" * 70)

        result = predictor.predict_single(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            verbose=not args.quiet
        )

        # Display results
        if not args.quiet:
            print("\n" + format_single_result(result, detailed=args.detailed))

        results = [result]

    else:
        # Batch mode
        tickers = args.tickers
        print(f"Mode: Batch prediction ({len(tickers)} stocks)")
        print(f"Tickers: {', '.join(tickers)}")
        print("=" * 70)

        results = predictor.predict_batch(
            tickers=tickers,
            start_date=args.start,
            end_date=args.end,
            verbose=not args.quiet
        )

        # Display batch summary
        if not args.quiet and len(results) > 0:
            print("\n" + format_batch_summary(results))

    # Save reports if requested
    if len(results) > 0:
        output_dir = Path(args.output_dir) if args.output_dir else None

        if args.save_csv:
            csv_path = export_results_to_csv(results, output_path=output_dir)
            if args.quiet:
                print(f"CSV saved to: {csv_path}")

        if args.save_json:
            json_path = export_results_to_json(results, output_path=output_dir)
            if args.quiet:
                print(f"JSON saved to: {json_path}")

    else:
        print("\nNo results to save (all predictions failed)")
        return 1

    print("\n✓ Prediction complete!")
    return 0


if __name__ == "__main__":
    exit(main())
