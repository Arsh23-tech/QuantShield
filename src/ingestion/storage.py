
from pathlib import Path

import pandas as pd


class MarketDataStorage:
    """Save and load QuantShield market datasets."""

    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, df, ticker):
        path = self.raw_dir / f"{ticker}_raw.parquet"
        df.to_parquet(path, index=False)
        return path

    def save_processed(self, df, ticker):
        path = self.processed_dir / f"{ticker}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load_processed(self, ticker):
        path = self.processed_dir / f"{ticker}.parquet"

        if not path.exists():
            raise FileNotFoundError(
                f"No processed data found for {ticker}: {path}"
            )

        return pd.read_parquet(path)
