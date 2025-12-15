"""Data downloaders for stock market data."""
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import yfinance as yf

from ..utils.paths import DATASETS_RAW_DIR


class YahooFinanceDownloader:
    """
    Downloads and caches stock data from Yahoo Finance.

    This class handles downloading historical stock data, caching it locally
    to avoid redundant downloads, and managing incremental updates.
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory to store downloaded data.
                     Defaults to DATASETS_RAW_DIR if not provided.
        """
        if data_dir is None:
            data_dir = DATASETS_RAW_DIR
        self.data_dir = Path(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get the file path for a given symbol and timeframe.

        Args:
            symbol: Stock ticker symbol
            timeframe: Timeframe (e.g., '1d', '1h')

        Returns:
            Path to the CSV file
        """
        filename = f"{symbol}_{timeframe}.csv"
        return self.data_dir / filename

    def _cache_covers_range(self, df: pd.DataFrame, start: datetime.date, end: datetime.date) -> bool:
        """
        Check if the cached dataset covers the requested date range.

        Args:
            df: Existing DataFrame
            start: Requested start date
            end: Requested end date

        Returns:
            True if cache covers the full range, False otherwise
        """
        # Ensure the index is DatetimeIndex for date checking
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index is not a DatetimeIndex. Please check CSV parsing logic.")

        cache_start = df.index.min().date()
        cache_end = df.index.max().date()
        print(f"Cache covers: {cache_start} to {cache_end}")

        # Check if cache covers the requested range
        covers_start = cache_start <= start
        covers_end = cache_end >= end
        is_valid = covers_start and covers_end

        if not is_valid:
            if not covers_start:
                print(f"Cache missing start: need {start}, have {cache_start}")
            if not covers_end:
                print(f"Cache missing end: need {end}, have {cache_end}")

        return is_valid

    def load_clean_csv(self, path: Path) -> pd.DataFrame:
        """
        Load and parse a CSV file with date parsing.

        Args:
            path: Path to CSV file

        Returns:
            DataFrame with DatetimeIndex
        """
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1d",
        force: bool = False
    ) -> pd.DataFrame:
        """
        Download data if needed and persist/return as DataFrame.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            timeframe: Interval for data (e.g., '1d', '1h')
            force: If True, always re-download even if cached

        Returns:
            DataFrame with stock data (OHLCV columns)

        Example:
            >>> downloader = YahooFinanceDownloader()
            >>> df = downloader.download('AAPL', '2020-01-01', '2020-12-31')
        """
        path = self._get_file_path(symbol, timeframe)
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()

        if os.path.exists(path) and not force:
            df = self.load_clean_csv(path)
            if self._cache_covers_range(df, start_date, end_date):
                # Filter cached data to requested date range
                df_filtered = df.loc[start_date:end_date]
                print(f"Using cached data: {df_filtered.shape[0]} rows ({start_date} to {end_date})")
                return df_filtered

        # Download from yfinance
        df = yf.download(symbol, start=start, end=end, interval=timeframe)
        print(f"Downloaded {df.shape[0]} rows from Yahoo Finance")

        if not df.empty:
            dataset = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                dataset.columns = dataset.columns.get_level_values(0)
            dataset.reset_index().to_csv(path, index=False)

        return df
