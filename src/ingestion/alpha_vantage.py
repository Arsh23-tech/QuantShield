
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingestion.base import MarketDataProvider


load_dotenv()


class AlphaVantageProvider(MarketDataProvider):
    """Fetch daily market prices from Alpha Vantage."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Missing API key. Set ALPHA_VANTAGE_API_KEY in your .env file."
            )

        self.session = requests.Session()

    def get_daily_prices(self, ticker, start_date, end_date):
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        response = self.session.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()

        # Alpha Vantage may return API messages instead of price data.
        if "Error Message" in payload:
            raise ValueError(payload["Error Message"])

        if "Note" in payload or "Information" in payload:
            message = payload.get("Note") or payload.get("Information")
            raise RuntimeError(f"Alpha Vantage API notice: {message}")

        series_key = "Time Series (Daily)"

        if series_key not in payload:
            raise ValueError(
                f"No daily price data returned for {ticker}. "
                f"Response keys: {list(payload.keys())}"
            )

        records = []

        for date, values in payload[series_key].items():
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": values["1. open"],
                    "high": values["2. high"],
                    "low": values["3. low"],
                    "close": values["4. close"],
                    "adjusted_close": values["5. adjusted close"],
                    "volume": values["6. volume"],
                }
            )

        df = pd.DataFrame(records)

        # Standardize types.
        df["date"] = pd.to_datetime(df["date"], errors="raise")

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]

        for column in price_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        df["volume"] = df["volume"].astype("int64")

        # Keep only the requested date range.
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        df = df.loc[df["date"].between(start, end)].copy()

        # Sort chronologically and return a clean index.
        df = df.sort_values("date").reset_index(drop=True)

        df["source"] = "alpha_vantage"
        df["ingested_at"] = datetime.now(timezone.utc)

        return df