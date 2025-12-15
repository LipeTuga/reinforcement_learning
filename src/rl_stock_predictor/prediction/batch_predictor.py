"""Batch prediction orchestrator for testing on multiple stocks."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from ..data.downloaders import YahooFinanceDownloader
from ..data.normalizers import normalize_df_zscore
from ..data.processors import add_technical_indicators
from ..environments.stock_trading import StockTradingEnv
from ..utils.device import get_device
from ..metrics.portfolio_metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
)
from ..metrics.trading_stats import calculate_trading_stats
from .benchmark import BuyAndHoldBenchmark
from .results import PredictionResult


class BatchPredictor:
    """
    Orchestrate predictions on single or multiple stocks.

    Features:
    - Single stock prediction with milestone reporting
    - Batch predictions on multiple tickers
    - Comprehensive performance metrics
    - Buy-and-hold benchmark comparison
    - Progress reporting at 25%, 50%, 75%, 100%

    Example:
        >>> predictor = BatchPredictor('models/ppo_multiasset_trader.zip')
        >>> result = predictor.predict_single('AAPL', '2024-01-01', '2024-12-31')
        >>> print(result)

        >>> results = predictor.predict_batch(
        ...     ['AAPL', 'MSFT', 'GOOGL'],
        ...     '2024-01-01',
        ...     '2024-12-31'
        ... )
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize batch predictor.

        Args:
            model_path: Path to trained RecurrentPPO model (.zip file)
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-detect device if needed
        if device == 'auto':
            device = get_device()
        self.device = device

        # Load model - try PPO first, then RecurrentPPO
        print(f"Loading model from {self.model_path}")
        print(f"Using device: {self.device}")

        # Detect model type by trying to load
        try:
            self.model = PPO.load(str(self.model_path), device=self.device)
            self.is_recurrent = False
            print("Model type: PPO (MLP)")
        except Exception:
            self.model = RecurrentPPO.load(str(self.model_path), device=self.device)
            self.is_recurrent = True
            print("Model type: RecurrentPPO (LSTM)")

        # Initialize downloader and benchmark
        self.downloader = YahooFinanceDownloader()
        self.benchmark = BuyAndHoldBenchmark(initial_balance=10000)

    def predict_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        verbose: bool = True
    ) -> PredictionResult:
        """
        Run prediction on a single stock with comprehensive metrics.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            verbose: Show progress milestones (25%, 50%, 75%, 100%)

        Returns:
            PredictionResult with all metrics and benchmark comparison

        Note:
            Uses environment's default initial_balance of $10,000
        """
        if verbose:
            print(f"\nPrediction: {ticker} ({start_date} to {end_date})")
            print("=" * 60)

        # Download data
        if verbose:
            print(f"Downloading data for {ticker}...")

        df = self.downloader.download(ticker, start_date, end_date)

        if df.empty:
            raise ValueError(f"No data found for {ticker} from {start_date} to {end_date}")

        # Add technical indicators (must match training data!)
        if verbose:
            print("Adding technical indicators...")
        df = add_technical_indicators(df)
        df = df.dropna()  # Drop rows with NaN from indicator calculations

        if df.empty:
            raise ValueError(f"Not enough data for {ticker} after adding indicators (need ~50 rows minimum)")

        # Create environment with normalization function
        # IMPORTANT: Pass original df + normalize function (not pre-normalized data!)
        # The environment needs original prices for net worth calculation
        env = StockTradingEnv(df=df, normalize_fn=normalize_df_zscore)

        # Initialize tracking
        obs, _ = env.reset()

        # For RecurrentPPO (LSTM)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        # Calculate milestone steps
        total_steps = len(df) - env.window_size
        milestones = {
            int(total_steps * 0.25): "25%",
            int(total_steps * 0.50): "50%",
            int(total_steps * 0.75): "75%",
            total_steps: "100%"
        }

        # Run prediction
        done = False
        step = 0

        while not done:
            # Predict action - different for PPO vs RecurrentPPO
            if self.is_recurrent:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_starts = np.array([done])
            step += 1

            # Report milestones
            if verbose and step in milestones:
                milestone = milestones[step]
                num_trades = sum(1 for a in env.actions if a != 0)
                print(f"Progress: {milestone:>4} | Net worth: ${info['net_worth']:>12,.2f} | "
                      f"Trades: {num_trades:>3}")

        # Get data from environment
        net_worth_history = env.net_worths
        action_history = env.actions
        cumulative_reward = env.cumulative_reward

        # Get price history from original dataframe
        price_history = df['Close'].iloc[
            env.window_size:env.window_size + len(net_worth_history)
        ].values.tolist()

        # Calculate returns
        returns = calculate_returns(net_worth_history)

        # Calculate performance metrics
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(net_worth_history)
        volatility = calculate_volatility(returns)
        sortino_ratio = calculate_sortino_ratio(returns)
        calmar_ratio = calculate_calmar_ratio(returns, max_drawdown)

        # Calculate trading statistics
        trading_stats = calculate_trading_stats(
            action_history,
            net_worth_history,
            price_history
        )

        # Run benchmark
        benchmark_results = self.benchmark.run(df)

        # Calculate model vs benchmark
        final_net_worth = net_worth_history[-1]
        model_profit_pct = ((final_net_worth - env.initial_balance) / env.initial_balance) * 100
        model_vs_benchmark = model_profit_pct - benchmark_results['profit_pct']

        # Create result object
        result = PredictionResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_balance=env.initial_balance,
            final_net_worth=final_net_worth,
            total_steps=total_steps,
            cumulative_reward=cumulative_reward,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=trading_stats['total_trades'],
            win_rate=trading_stats['win_rate'],
            profit_factor=trading_stats['profit_factor'],
            avg_holding_period=trading_stats['avg_holding_period'],
            num_buys=trading_stats['num_buys'],
            num_sells=trading_stats['num_sells'],
            buy_hold_final_value=benchmark_results['final_value'],
            buy_hold_profit_pct=benchmark_results['profit_pct'],
            model_vs_benchmark=model_vs_benchmark,
            net_worth_history=net_worth_history,
            action_history=action_history,
            price_history=price_history,
        )

        if verbose:
            print("\n" + "=" * 60)
            print(result)
            print("=" * 60)

        return result

    def predict_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        verbose: bool = True,
        save_report: bool = False
    ) -> List[PredictionResult]:
        """
        Run predictions on multiple stocks.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            verbose: Show progress for each stock
            save_report: Save results to CSV/JSON (requires reporting module)

        Returns:
            List of PredictionResult objects, one per ticker
        """
        print(f"\nBatch Prediction: {len(tickers)} stocks")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 60)

        results = []

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

            try:
                result = self.predict_single(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=verbose
                )
                results.append(result)

            except Exception as e:
                print(f"ERROR: Failed to predict {ticker}: {e}")
                continue

        # Print summary
        if results:
            self._print_batch_summary(results)

        # Save report if requested
        if save_report:
            print("\nNote: Report saving requires reporting module (not yet implemented)")

        return results

    def _print_batch_summary(self, results: List[PredictionResult]) -> None:
        """Print summary table of batch prediction results."""
        print("\n" + "=" * 60)
        print("BATCH PREDICTION SUMMARY")
        print("=" * 60)

        # Create summary DataFrame
        summary_df = pd.DataFrame([r.to_series() for r in results])

        # Display key columns
        display_cols = [
            'ticker',
            'profit_pct',
            'buy_hold_profit_pct',
            'model_vs_benchmark',
            'sharpe_ratio',
            'max_drawdown',
            'total_trades',
            'win_rate'
        ]

        print("\nPerformance Summary:")
        print(summary_df[display_cols].to_string(index=False))

        # Best/worst performers
        best = results[summary_df['profit_pct'].idxmax()]
        worst = results[summary_df['profit_pct'].idxmin()]

        print(f"\nBest performer:  {best.ticker} ({best.profit_pct:+.2f}%)")
        print(f"Worst performer: {worst.ticker} ({worst.profit_pct:+.2f}%)")

        # Average metrics
        avg_model_return = summary_df['profit_pct'].mean()
        avg_benchmark_return = summary_df['buy_hold_profit_pct'].mean()
        avg_outperformance = summary_df['model_vs_benchmark'].mean()

        print(f"\nAverage model return:     {avg_model_return:+.2f}%")
        print(f"Average benchmark return: {avg_benchmark_return:+.2f}%")
        print(f"Average outperformance:   {avg_outperformance:+.2f}%")

        # Count stocks that beat benchmark
        beat_benchmark = (summary_df['model_vs_benchmark'] > 0).sum()
        print(f"\nStocks beating benchmark: {beat_benchmark}/{len(results)}")

        print("=" * 60)
