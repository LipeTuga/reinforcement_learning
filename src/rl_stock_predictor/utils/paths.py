"""Path management for the RL stock predictor project."""
from pathlib import Path

# Project root directory (4 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_RAW_DIR = DATASETS_DIR / "raw"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_ARCHIVE_DIR = MODELS_DIR / "archive"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Experiments directory
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure all directories exist
for directory in [
    DATASETS_DIR,
    DATASETS_RAW_DIR,
    MODELS_DIR,
    MODELS_ARCHIVE_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    TENSORBOARD_DIR,
    EXPERIMENTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
