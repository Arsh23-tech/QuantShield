
from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TICKERS,
    START_DATE,
    END_DATE,
)

from src.ingestion.yfinance_provider import YFinanceProvider
from src.ingestion.storage import MarketDataStorage
from src.validation.market_validator import validate_market_data
import pandas as pd

def main():
    provider = YFinanceProvider()

    storage = MarketDataStorage(
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
    )

    results = []

    for ticker in TICKERS:
        print(f"\nDownloading {ticker}...")

        try:
            prices = provider.get_daily_prices(
                ticker=ticker,
                start_date=START_DATE,
                end_date=END_DATE,
            )

            report = validate_market_data(prices)

            if report["status"] != "PASS":
                print(f"Validation failed for {ticker}: {report}")
                results.append((ticker, "FAIL", len(prices)))
                continue

            storage.save_processed(prices, ticker)

            print(f"Saved {len(prices)} rows for {ticker}")
            results.append((ticker, "PASS", len(prices)))

        except Exception as exc:
            print(f"Error processing {ticker}: {exc}")
            results.append((ticker, "ERROR", 0))

    print("\nQUANTSHIELD INGESTION SUMMARY")
    for ticker, status, rows in results:
        print(f"{ticker}: {status} | {rows} rows")

    successful_tickers = [
        ticker
        for ticker, status, rows in results
        if status == "PASS"
    ]

    if successful_tickers:
        frames = [
            storage.load_processed(ticker)
            for ticker in successful_tickers
        ]

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        combined_path = PROCESSED_DATA_DIR / "market_prices.parquet"
        combined.to_parquet(combined_path, index=False)

        print(f"\nCombined dataset saved: {combined_path}")
        print(f"Total rows: {len(combined)}")
        print(f"Unique tickers: {combined['ticker'].nunique()}")
        print(
            f"Date range: {combined['date'].min()} "
            f"to {combined['date'].max()}"
        )

if __name__ == "__main__":
    main()
