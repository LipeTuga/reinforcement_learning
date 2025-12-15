# Quick Start - Optimized for M4 MacBook Pro

## 🚀 Ready to Train!

Your project is now optimized for **maximum performance in ~20 minutes**!

## Default Configuration

Just run:
```bash
python scripts/train.py
```

### What it does:
- **Stocks**: 8 major tech stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX)
- **Timesteps**: 100,000
- **Device**: MPS (Apple Silicon GPU) - auto-detected
- **Data range**: 2000-2024 (24 years of data)
- **Expected time**: ~20 minutes on M4 MacBook Pro
- **Output**: Trained model saved to `models/ppo_multiasset_trader.zip`

## Training Configurations

### Under 20 Minutes (Default)
```bash
# 8 stocks, 100k timesteps (~20 min)
python scripts/train.py
```

### Quick Test Run (~5 min)
```bash
# 4 stocks, 50k timesteps
python scripts/train.py \
    --tickers AAPL MSFT GOOGL AMZN \
    --timesteps 50000
```

### Maximum Within 20 Minutes
```bash
# 10 stocks, 100k timesteps (~23 min - slightly over)
python scripts/train.py \
    --tickers AAPL MSFT GOOGL AMZN TSLA NVDA META NFLX AMD CRM \
    --timesteps 100000
```

### Longer Training for Better Results
```bash
# 8 stocks, 200k timesteps (~40 min)
python scripts/train.py --timesteps 200000
```

## Expected Training Times

| Stocks | Timesteps | Device | Time |
|--------|-----------|--------|------|
| 8 | 50k | MPS | ~10 min |
| 8 | 100k | MPS | **~20 min** ⭐ |
| 8 | 150k | MPS | ~30 min |
| 8 | 200k | MPS | ~40 min |
| 8 | 100k | CPU | ~4h |

## Monitor Training Progress

### Watch GPU Usage
```bash
# Terminal 1: Run training
python scripts/train.py

# Terminal 2: Monitor GPU
sudo powermetrics --samplers gpu_power -i 1000
```

### Check Live Plots
Training plots are saved every 1000 steps to:
```
outputs/plots/training_plot_live.png
```

Open in Preview and it will auto-update:
```bash
open outputs/plots/training_plot_live.png
```

## After Training

### Run Predictions
```bash
# Test on a different stock (Microsoft)
python scripts/predict.py --ticker MSFT

# Test on a stock not in training (Tesla)
python scripts/predict.py --ticker TSLA

# Results saved to: outputs/plots/net_worth.png
```

### View Results
```bash
open outputs/plots/net_worth.png
```

## Tips for Best Results

1. **Let it finish**: Don't interrupt training, especially near the end
2. **Plug in to power**: GPU runs faster when not on battery
3. **Close other apps**: Free up GPU resources
4. **Monitor temperature**: Keep laptop cool for sustained performance

## Customize Your Training

### More Stocks
```bash
python scripts/train.py \
    --tickers AAPL MSFT GOOGL AMZN TSLA NVDA META NFLX JPM BAC WFC GS
```

### Different Time Period
```bash
# Focus on recent data (faster, more relevant)
python scripts/train.py \
    --start 2020-01-01 \
    --end 2024-12-31
```

### Continue Training
```bash
# Resume from existing model
python scripts/train.py --continue-training --timesteps 50000
```

### Custom Model Name
```bash
python scripts/train.py \
    --model-name my_experiment_v1 \
    --timesteps 100000
```

## Verify GPU is Working

Before your first training run:
```bash
python -c "from rl_stock_predictor.utils.device import get_device, print_device_info; print_device_info(get_device())"
```

You should see:
```
============================================================
Device Configuration
============================================================
Selected device: MPS
Backend: Metal Performance Shaders (Apple Silicon)
MPS available: True
MPS built: True
============================================================
```

## What You'll Get

After training completes:
1. **Trained model**: `models/ppo_multiasset_trader.zip`
2. **Training plots**: `outputs/plots/training_plot_live.png`
3. **Training logs**: `logs/RecurrentPPO_*/`
4. **Downloaded data**: `datasets/raw/*.csv` (cached for reuse)

## Next Steps

1. **Test the model**: `python scripts/predict.py`
2. **Try different stocks**: Experiment with sector-specific portfolios
3. **Tune hyperparameters**: Adjust learning rate, batch size, etc.
4. **Longer training**: 200k-500k timesteps for production models

## Troubleshooting

### Training slower than expected?
- Verify MPS is enabled: Check device detection output
- Close other GPU-heavy apps: Chrome, video editors, etc.
- Plug in to power: Battery mode throttles GPU

### Out of memory?
- Reduce number of stocks: Try 4-6 stocks instead of 8
- Use shorter date range: `--start 2020-01-01` instead of 2000

### Want to experiment?
- Lower timesteps for faster iteration: `--timesteps 25000` (~5 min)
- Use single stock for debugging: `--tickers AAPL`

---

**Ready to train? Just run:**
```bash
python scripts/train.py
```

**Training will start in ~20 minutes! ⚡**
