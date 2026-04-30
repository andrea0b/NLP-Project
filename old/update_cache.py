"""
Download and cache historical stock prices. Run once, then never again.

Usage:
    python update_cache.py              # Cache prices for data/ (batch_size=15)
    python update_cache.py <dir>        # Cache prices for <dir>
    python update_cache.py <dir> <size> # Custom batch size (e.g., 20)

This downloads prices from yfinance and stores them as prices_cache.parquet.
After running, get_processed_data() will use cached prices (no network calls).
"""

import sys
from pathlib import Path
from data_cleaningv3 import load_raw_data, update_prices_cache

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    data_dir = Path(data_dir)

    print("=" * 70)
    print("DOWNLOAD & CACHE HISTORICAL PRICES")
    print("=" * 70)
    print(f"\nLoading articles from {data_dir}...")
    raw_df = load_raw_data(data_dir)
    print(f"Found {len(raw_df)} articles")
    print(f"Unique tickers to download: {raw_df['mentioned_companies'].explode().nunique()}")

    print(f"\n📥 Downloading prices (batch_size={batch_size})...\n")
    update_prices_cache(raw_df, data_dir, batch_size=batch_size)

    print("\n" + "=" * 70)
    print("✓ DONE! Prices are now cached.")
    print("=" * 70)
    print("\nYou can now run get_processed_data() without network delays:")
    print("  from utils.data_cleaningv3 import get_processed_data")
    print("  df = get_processed_data('data', force_refresh=True)")

if __name__ == "__main__":
    main()
