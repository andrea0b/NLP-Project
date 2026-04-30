import json
import lzma
from pathlib import Path

import pandas as pd


KEEP_COLS = [
    "title",
    "maintext",
    "date_publish",
    "mentioned_companies",
    "sentiment",
    "emotion",
    "named_entities",
]


def _load_json_xz(path: Path) -> pd.DataFrame:
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        return pd.DataFrame(json.load(fh))


def _extract_ticker_prices(row: pd.Series) -> list[dict]:
    tickers = row.get("mentioned_companies")
    if not isinstance(tickers, list) or len(tickers) == 0:
        return []

    ticker_rows = []
    for ticker in tickers:
        prev_price = row.get(f"prev_day_price_{ticker}")
        curr_price = row.get(f"curr_day_price_{ticker}")
        next_price = row.get(f"next_day_price_{ticker}")

        if pd.notna(prev_price) and pd.notna(next_price):
            ticker_rows.append(
                {
                    "ticker": ticker,
                    "prev_price": prev_price,
                    "curr_price": curr_price,
                    "next_price": next_price,
                }
            )

    return ticker_rows


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sentiment_df = pd.json_normalize(
        df["sentiment"].apply(lambda x: x if isinstance(x, dict) else {})
    ).add_prefix("raw_sentiment_")

    emotion_df = pd.json_normalize(
        df["emotion"].apply(lambda x: x if isinstance(x, dict) else {})
    ).add_prefix("raw_emotion_")

    df = pd.concat(
        [df.drop(columns=["sentiment", "emotion"]), sentiment_df, emotion_df],
        axis=1,
    )

    for col in sentiment_df.columns.tolist() + emotion_df.columns.tolist():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_and_clean_data(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load all .json.xz files and return a ticker-expanded, cleaned dataframe."""
    files = sorted(Path(data_dir).glob("*.json.xz"))
    if not files:
        raise FileNotFoundError(f"No .json.xz files found in: {Path(data_dir).resolve()}")

    df = pd.concat((_load_json_xz(path) for path in files), ignore_index=True)

    ticker_data = df.apply(_extract_ticker_prices, axis=1).explode().dropna()
    if ticker_data.empty:
        df_with_tickers = pd.DataFrame(
            columns=KEEP_COLS + ["ticker", "prev_price", "curr_price", "next_price"]
        )
    else:
        ticker_df = pd.DataFrame(ticker_data.tolist(), index=ticker_data.index)
        df_with_tickers = (
            df.loc[ticker_df.index, KEEP_COLS]
            .reset_index(drop=True)
            .join(ticker_df.reset_index(drop=True))
        )

    return _clean_cols(df_with_tickers)
