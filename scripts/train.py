"""Training script for RL stock trading model."""
import os
import argparse
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from dotenv import load_dotenv

from rl_stock_predictor.callbacks.live_logger import LiveLoggerCallback
from rl_stock_predictor.environments.stock_trading import StockTradingEnv
from rl_stock_predictor.data.downloaders import YahooFinanceDownloader
from rl_stock_predictor.data.normalizers import normalize_df_zscore
from rl_stock_predictor.data.processors import align_dataframes
from rl_stock_predictor.utils.paths import MODELS_DIR

load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL stock trading model")
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=["AAPL", "EOG"],
        help="List of stock tickers to train on"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1990-01-01",
        help="Start date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="End date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_multiasset_trader",
        help="Name for the model (saved in models/ directory)"
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1000,
        help="Generate plots every N training steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cpu/cuda). Defaults to DEVICE env var"
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue training from existing model if it exists"
    )
    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # Determine device
    device = args.device or os.getenv("DEVICE", "cpu")

    print(f"Training configuration:")
    print(f"  Tickers: {args.tickers}")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Device: {device}")
    print(f"  Model name: {args.model_name}")
    print()

    # Download the data
    print("Downloading data...")
    downloader = YahooFinanceDownloader()
    dfs = [downloader.download(ticker, args.start, args.end) for ticker in args.tickers]

    # Prepare the data - align to common dates
    print("Aligning datasets to common dates...")
    aligned_dfs = align_dataframes(dfs)

    # Prepare one environment per ticker for DummyVecEnv
    def make_env(df):
        return lambda: StockTradingEnv(df, normalize_df_zscore)

    env_fns = [make_env(df) for df in aligned_dfs]
    vec_env = DummyVecEnv(env_fns)
    callback = LiveLoggerCallback(plot_every=args.plot_every)

    # Train the model
    model_path = MODELS_DIR / args.model_name

    if args.continue_training and os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}.zip for continued training.")
        model = RecurrentPPO.load(model_path, env=vec_env, device=device)
    else:
        print("Creating new model.")
        model = RecurrentPPO("MlpLstmPolicy", env=vec_env, verbose=1, device=device)

    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save the model
    print(f"Saving model to {model_path}.zip...")
    model.save(model_path)

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}.zip")
    print(f"Training plots saved to: outputs/plots/training_plot_live.png")

    # Inference: predictions per ticker in parallel
    print("\nRunning inference on training data...")
    obs = vec_env.reset()
    terminated = [False] * len(args.tickers)
    truncated = [False] * len(args.tickers)

    while not all(terminated) and not all(truncated):
        actions, _ = model.predict(obs, deterministic=True)
        step_result = vec_env.step(actions)

        if len(step_result) == 5:
            obs, rewards, term, trunc, infos = step_result
            terminated = [a or b for a, b in zip(terminated, term)]
            truncated = [a or b for a, b in zip(truncated, trunc)]
        elif len(step_result) == 4:
            obs, rewards, done, infos = step_result
            terminated = [a or b for a, b in zip(terminated, done)]
            truncated = [False] * len(done)
        else:
            raise RuntimeError("Unexpected number of values returned by vec_env.step()")

    print("Inference complete!")


if __name__ == "__main__":
    main()
