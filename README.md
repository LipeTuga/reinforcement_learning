# RL Stock Predictor

A Reinforcement Learning-based stock trading system that trains AI agents to make buy/hold/sell decisions using deep reinforcement learning.

## Overview

This project uses **Proximal Policy Optimization (PPO)** with LSTM neural networks to learn profitable trading strategies from historical stock data. The agent observes a sliding window of stock prices (OHLCV data) and learns to maximize portfolio value through trading actions.

### Key Features

- **Recurrent PPO (RecurrentPPO)**: Uses LSTM to capture temporal dependencies in price movements
- **Multi-asset Training**: Train on multiple stocks simultaneously
- **Live Monitoring**: Real-time training metrics and visualization
- **Risk Management**: Built-in penalties for holding stocks too long
- **Modular Design**: Clean separation of data, environments, training, and inference
- **Flexible Data Pipeline**: Easy integration with Yahoo Finance

## Installation

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rl-stock-predictor
```

2. Install dependencies:
```bash
poetry install
```

Or with pip:
```bash
pip install -e .
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env to set DEVICE=cuda if you have a GPU
```

## Quick Start

### Training a Model

Train on default stocks (AAPL, EOG) from 1990-2024:

```bash
python scripts/train.py
```

Train with custom parameters:

```bash
python scripts/train.py \
    --tickers AAPL MSFT GOOGL \
    --start 2010-01-01 \
    --end 2023-12-31 \
    --timesteps 100000 \
    --model-name my_custom_model
```

### Running Predictions

Run inference on a different stock:

```bash
python scripts/predict.py --ticker TSLA
```

With custom model:

```bash
python scripts/predict.py \
    --ticker NVDA \
    --model-name my_custom_model \
    --start 2020-01-01 \
    --end 2024-12-31
```

## Project Structure

```
rl-stock-predictor/
├── scripts/              # Executable training and prediction scripts
│   ├── train.py          # Training script with CLI
│   └── predict.py        # Prediction script with CLI
│
├── src/rl_stock_predictor/  # Library code (importable)
│   ├── data/             # Data handling
│   │   ├── downloaders.py    # Yahoo Finance downloader
│   │   ├── normalizers.py    # Data normalization functions
│   │   └── processors.py     # Data alignment and processing
│   ├── environments/     # RL environments
│   │   └── stock_trading.py  # Stock trading Gym environment
│   ├── callbacks/        # Training callbacks
│   │   └── live_logger.py    # Live training monitor
│   ├── utils/            # Shared utilities
│   │   └── paths.py          # Path management
│   └── config/           # Configuration management
│
├── models/               # Trained models (gitignored)
├── datasets/             # Downloaded stock data (gitignored)
├── logs/                 # Training logs (gitignored)
├── outputs/              # Generated plots and reports (gitignored)
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit and integration tests
└── docs/                 # Additional documentation
```

## How It Works

### 1. Environment

The `StockTradingEnv` is a custom Gym environment that simulates stock trading:

- **Action Space**: Discrete(3) - Hold (0), Buy (1), Sell (2)
- **Observation Space**: 14-day sliding window of normalized OHLCV data
- **Reward Function**: Based on net worth changes with penalties for holding too long
- **Starting Balance**: $10,000

### 2. Agent

Uses RecurrentPPO with LSTM policy:
- **Policy**: MlpLstmPolicy (LSTM + MLP layers)
- **Algorithm**: Proximal Policy Optimization
- **Sequential Learning**: LSTM captures time-series patterns

### 3. Training Pipeline

1. **Data Collection**: Downloads historical data from Yahoo Finance
2. **Data Alignment**: Aligns multiple stocks to common trading days
3. **Normalization**: Z-score normalization for stable training
4. **Vectorized Environments**: Parallel environments for multi-asset training
5. **Training**: PPO learns through trial and error
6. **Monitoring**: Live plots of net worth, rewards, trades

### 4. Inference

The trained model can predict on new stocks or future time periods:
- Loads trained weights
- Runs the environment in evaluation mode
- Generates trading decisions deterministically
- Visualizes net worth trajectory

## Usage Examples

### Using as a Library

```python
from rl_stock_predictor import (
    StockTradingEnv,
    YahooFinanceDownloader,
    normalize_df_zscore,
    align_dataframes
)
from sb3_contrib import RecurrentPPO

# Download data
downloader = YahooFinanceDownloader()
df = downloader.download('AAPL', '2020-01-01', '2023-12-31')

# Create environment
env = StockTradingEnv(df, normalize_df_zscore)

# Train model
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("my_model")
```

### Custom Normalization

```python
from rl_stock_predictor.data.normalizers import (
    normalize_df_zscore,      # Z-score normalization
    normalize_df_minmax,      # Min-max scaling
    normalize_df_robust       # Robust scaler (IQR-based)
)

env = StockTradingEnv(df, normalize_df_minmax)
```

## Training Metrics

During training, the following metrics are tracked and plotted:

- **Net Worth**: Total portfolio value over time
- **Cumulative Reward**: Sum of rewards received
- **Number of Trades**: Count of buy/sell actions
- **Portfolio Value**: Current cash + stock holdings value

Plots are saved to `outputs/plots/training_plot_live.png` and updated every N steps.

## CLI Reference

### train.py

```
--tickers          List of stock tickers (default: AAPL EOG)
--start            Start date YYYY-MM-DD (default: 1990-01-01)
--end              End date YYYY-MM-DD (default: 2024-01-01)
--timesteps        Total training steps (default: 50000)
--model-name       Model name (default: ppo_multiasset_trader)
--plot-every       Plot frequency (default: 1000)
--device           cpu or cuda (default: from .env)
--continue-training  Resume from existing model
```

### predict.py

```
--ticker           Stock ticker to predict (default: MSFT)
--start            Start date YYYY-MM-DD (default: 1990-01-01)
--end              End date YYYY-MM-DD (default: 2025-03-01)
--model-name       Model to load (default: ppo_multiasset_trader)
--device           cpu or cuda (default: from .env)
--render           Save plot (default: True)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/ scripts/ tests/

# Lint
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### Adding New Environments

1. Create new environment in `src/rl_stock_predictor/environments/`
2. Inherit from `gym.Env`
3. Implement `step()`, `reset()`, `_next_observation()`, `render()`
4. Export in `environments/__init__.py`

### Adding New Data Sources

1. Create downloader in `src/rl_stock_predictor/data/downloaders.py`
2. Follow the `YahooFinanceDownloader` interface
3. Export in `data/__init__.py`

## Performance Tips

1. **Use GPU**: Set `DEVICE=cuda` in `.env` for 5-10x faster training
2. **Increase Timesteps**: More training = better performance (100k+ recommended)
3. **Multi-Asset Training**: Training on multiple stocks improves generalization
4. **Feature Engineering**: Add technical indicators in `data/processors.py`
5. **Hyperparameter Tuning**: Experiment with learning rate, batch size, etc.

## Troubleshooting

### Import Errors

If you see import errors, ensure the package is installed:
```bash
pip install -e .
```

### CUDA Out of Memory

Reduce the number of parallel environments or use CPU:
```bash
export DEVICE=cpu
```

### Data Download Issues

Yahoo Finance sometimes has rate limits. Add delays or use cached data:
```python
downloader.download('AAPL', '2020-01-01', '2024-01-01', force=False)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Data from [Yahoo Finance](https://finance.yahoo.com)
- Environment follows [Gymnasium](https://gymnasium.farama.org/) API

## Future Improvements

See `docs/tasks.md` for a comprehensive list of planned improvements, including:

- Portfolio management with multiple assets simultaneously
- Options and derivatives trading
- Advanced technical indicators
- Backtesting framework
- Live trading integration
- Web dashboard for monitoring
- More sophisticated reward functions

## Contact

[Add your contact information]
