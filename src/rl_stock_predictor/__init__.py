"""RL Stock Predictor - Reinforcement Learning for Stock Trading."""

# Import key classes for easy access
from .environments.stock_trading import StockTradingEnv
from .data.downloaders import YahooFinanceDownloader
from .callbacks.live_logger import LiveLoggerCallback
from .data.normalizers import normalize_df_zscore, normalize_df_minmax, normalize_df_robust
from .data.processors import (
    align_dataframes,
    create_train_test_split,
    add_technical_indicators,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)

# Prediction and benchmarking
from .prediction import BatchPredictor, PredictionResult, BuyAndHoldBenchmark

# Metrics
from .metrics.portfolio_metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
)
from .metrics.trading_stats import calculate_trading_stats

# Reporting
from .reporting import (
    export_results_to_csv,
    export_results_to_json,
    format_single_result,
    format_batch_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "StockTradingEnv",
    "YahooFinanceDownloader",
    "LiveLoggerCallback",
    # Data processing
    "normalize_df_zscore",
    "normalize_df_minmax",
    "normalize_df_robust",
    "align_dataframes",
    "create_train_test_split",
    # Technical indicators
    "add_technical_indicators",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    # Prediction
    "BatchPredictor",
    "PredictionResult",
    "BuyAndHoldBenchmark",
    # Portfolio metrics
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_volatility",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    # Trading stats
    "calculate_trading_stats",
    # Reporting
    "export_results_to_csv",
    "export_results_to_json",
    "format_single_result",
    "format_batch_summary",
]
