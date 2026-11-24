
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.ingestion.base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    """Fetch daily market data for QuantShield research."""

    def get_daily_prices(self, ticker, start_date, end_date):
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            multi_level_index=False,
        )

        if df.empty:
            raise ValueError(f"No market data returned for {ticker}")

        # Normalize provider column names.
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )

        required = [
            "open", "high", "low", "close",
            "adjusted_close", "volume",
        ]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        df = df.reset_index()
        df["ticker"] = ticker

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="raise")

        df["volume"] = df["volume"].astype("int64")
        df["source"] = "yfinance"
        df["ingested_at"] = datetime.now(timezone.utc)

        columns = [
            "date", "ticker", "open", "high", "low",
            "close", "adjusted_close", "volume",
            "source", "ingested_at",
        ]

        return (
            df[columns]
            .sort_values("date")
            .reset_index(drop=True)
        )