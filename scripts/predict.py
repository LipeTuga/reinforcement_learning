"""Prediction/inference script for trained RL stock trading model."""
import os
import argparse
from dotenv import load_dotenv
from sb3_contrib import RecurrentPPO

from rl_stock_predictor.environments.stock_trading import StockTradingEnv
from rl_stock_predictor.data.downloaders import YahooFinanceDownloader
from rl_stock_predictor.data.normalizers import normalize_df_zscore
from rl_stock_predictor.utils.paths import MODELS_DIR

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with trained RL stock trading model")
    parser.add_argument(
        "--ticker",
        type=str,
        default="MSFT",
        help="Stock ticker to predict on"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1990-01-01",
        help="Start date for prediction data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-03-01",
        help="End date for prediction data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_multiasset_trader",
        help="Name of the model to load (from models/ directory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cpu/cuda). Defaults to DEVICE env var"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render the results (save plot)"
    )
    return parser.parse_args()


def main():
    """Main prediction loop."""
    args = parse_args()

    # Determine device
    device = args.device or os.getenv("DEVICE", "cpu")

    print(f"Prediction configuration:")
    print(f"  Ticker: {args.ticker}")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {device}")
    print()

    # Download data
    print("Downloading data...")
    downloader = YahooFinanceDownloader()
    partial_df = downloader.download(symbol=args.ticker, start=args.start, end=args.end)

    # Create environment
    env = StockTradingEnv(partial_df, normalize_df_zscore)
    obs, _ = env.reset()

    # Load model
    model_path = MODELS_DIR / args.model_name
    print(f"Loading model from {model_path}.zip...")
    model = RecurrentPPO.load(model_path, device=device)

    # Run prediction
    print("Running prediction...")
    terminated = False
    truncated = False
    step_count = 0

    while not truncated and not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, results = env.step(action)
        step_count += 1

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"  Step {step_count}: Net worth = ${results['net_worth']:.2f}")

    # Final results
    print("\nPrediction complete!")
    print(f"  Total steps: {step_count}")
    print(f"  Final net worth: ${results['net_worth']:.2f}")
    print(f"  Initial balance: $10,000.00")
    print(f"  Profit/Loss: ${results['net_worth'] - 10000:.2f} ({(results['net_worth'] / 10000 - 1) * 100:.2f}%)")
    print(f"  Cumulative reward: {results['cumulative_reward']:.2f}")

    # Render results
    if args.render:
        print("\nRendering results...")
        env.render()
        print(f"Plot saved to: outputs/plots/net_worth.png")


if __name__ == "__main__":
    main()
