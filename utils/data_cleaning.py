import logging
from io import StringIO
from contextlib import redirect_stdout

logging.basicConfig(level=logging.CRITICAL, force=True)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('fastcoref').setLevel(logging.ERROR)

import json
import lzma
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import torch

neural_coref_model = None
neural_coref_device = 'disabled'
_coref_initialized = False

def _init_neural_coref():
    global neural_coref_model, neural_coref_device, _coref_initialized
    if _coref_initialized:
        return
    _coref_initialized = True

    try:
        from fastcoref import FCoref
        from fastcoref.modeling import FCorefModel
        import os
        import sys

        # Patch 1: Add missing all_tied_weights_keys for transformers 5.x
        if not hasattr(FCorefModel, 'all_tied_weights_keys'):
            def get_tied_weights(self):
                tied = getattr(self, '_tied_weights_keys', None)
                return tied if tied is not None else {}
            FCorefModel.all_tied_weights_keys = property(get_tied_weights)

        # Patch 2: Suppress FCoref LOAD REPORT and other output
        import warnings
        warnings.filterwarnings('ignore')

        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        try:
            with redirect_stdout(StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                neural_coref_model = FCoref()
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

        neural_coref_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        neural_coref_device = 'disabled'
    except Exception as e:
        print(f"[WARN] Neural coref init failed: {e}", file=sys.stderr)
        neural_coref_device = 'disabled'

_init_neural_coref()

with redirect_stdout(StringIO()):
    import spacy
    from transformers import AutoTokenizer

try:
    with redirect_stdout(StringIO()):
        nlp = spacy.load("en_core_web_sm")
except OSError:
    try:
        with redirect_stdout(StringIO()):
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

finbert_tokenizer = None
try:
    with redirect_stdout(StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
except Exception:
    pass

TARGET_TOKEN = "TGT_CO"
OTHER_TOKEN = "OTHER_CO"
DEFAULT_THRESHOLD = 0.005
CACHE_FILENAME = "processed_dataset.parquet"
MAPPING_FILENAME = "company_mapping.json"
PRICES_CACHE_FILENAME = "prices_cache.parquet"
VOLATILITY_CACHE_FILENAME = "volatility_thresholds.json"

def apply_neural_coref(text: str) -> str:
    if not neural_coref_model or not isinstance(text, str) or not text.strip():
        return text
    try:
        preds = neural_coref_model.predict(texts=[text])
        coref_result = preds[0]
        clusters = coref_result.get_clusters()

        if not clusters:
            return text

        # Build replacements from highest span to lowest (to preserve indices)
        replacements = []
        for cluster in clusters:
            if len(cluster) > 1:
                # Get referent text (first mention)
                referent_mention = cluster[0]
                referent_text = referent_mention.text if hasattr(referent_mention, 'text') else str(referent_mention)

                # Replace other mentions with referent
                for mention in cluster[1:]:
                    mention_text = mention.text if hasattr(mention, 'text') else str(mention)
                    # Only replace if it's a pronoun or short mention (avoid replacing proper nouns)
                    if mention_text.lower() in ['it', 'its', 'itself', 'they', 'their', 'them', 'this', 'that']:
                        replacements.append((mention_text, referent_text))

        # Apply replacements (case-insensitive)
        resolved_text = text
        for mention_text, referent_text in replacements:
            resolved_text = re.sub(rf'\b{re.escape(mention_text)}\b', referent_text, resolved_text, flags=re.IGNORECASE)

        return resolved_text
    except Exception:
        return text

def load_company_mapping(mapping_path: str | Path) -> Dict[str, List[str]]:
    path = Path(mapping_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_raw_data(data_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.json.xz"))
    all_dfs = []
    for f in files:
        with lzma.open(f, "rt", encoding="utf-8") as fh:
            all_dfs.append(pd.DataFrame(json.load(fh)))
    df = pd.concat(all_dfs, ignore_index=True)
    critical_cols = ['title', 'maintext', 'date_publish']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    return df

def get_yearly_volatility_thresholds(data_dir: str | Path, df: pd.DataFrame, min_threshold: float = 0.005) -> Dict[str, Dict[str, float]]:
    """Load volatility thresholds from cache. Falls back to computing from prices if cache missing."""
    import json
    data_dir = Path(data_dir)
    volatility_cache = {}

    # Try to load from JSON cache first
    json_cache_path = data_dir / VOLATILITY_CACHE_FILENAME
    if json_cache_path.exists():
        try:
            with open(json_cache_path) as f:
                cached_vol = json.load(f)
            # Convert string years to int keys
            for ticker, years_dict in cached_vol.items():
                volatility_cache[ticker] = {int(year): float(vol) for year, vol in years_dict.items()}
            return volatility_cache
        except Exception:
            pass

    # Fallback: compute from prices cache if JSON cache not available
    prices_cache_path = data_dir / PRICES_CACHE_FILENAME
    if not prices_cache_path.exists():
        return volatility_cache

    try:
        prices_df = pd.read_parquet(prices_cache_path)
        prices_df['date'] = pd.to_datetime(prices_df['date'])

        # Identify required ticker-years
        required_pairs = set()
        for _, row in df.iterrows():
            tickers = row.get("mentioned_companies", [])
            if not isinstance(tickers, list): continue
            try:
                year = int(pd.to_datetime(row['date_publish']).year)
                for t in tickers:
                    required_pairs.add((t, year))
            except Exception:
                continue

        # Compute volatility for each required pair
        for ticker, year in required_pairs:
            if ticker not in volatility_cache:
                volatility_cache[ticker] = {}

            vol = _compute_volatility(prices_df, ticker, year, min_threshold)
            volatility_cache[ticker][year] = vol
    except Exception:
        pass

    return volatility_cache

def _fetch_and_cache_prices(tickers: List[str], min_year: int, max_year: int, cache_path: Path) -> pd.DataFrame:
    """Batch download prices for tickers and cache as parquet."""
    try:
        import yfinance as yf
        with redirect_stdout(StringIO()):
            prices = yf.download(tickers, start=f"{min_year}-01-01", end=f"{max_year}-12-31", progress=False)

        if prices is None or prices.empty:
            return pd.DataFrame()

        # yfinance returns multi-index columns like ('Close', 'AAPL') - extract as {ticker: [prices]}
        result = []
        for ticker in tickers:
            # Find the 'Close' or 'Adj Close' column for this ticker
            close_col = None
            for col in prices.columns:
                if isinstance(col, tuple) and ticker in col and ('Close' in col or 'Adj' in col):
                    close_col = col
                    break

            if close_col is None:
                continue

            prices_data = prices[[close_col]].reset_index()
            prices_data.columns = ['date', 'price']
            prices_data['ticker'] = ticker
            result.append(prices_data)

        if result:
            cached = pd.concat(result, ignore_index=True)
            cached.to_parquet(cache_path, index=False)
            return cached
        return pd.DataFrame()
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()

def _compute_volatility(prices_df: pd.DataFrame, ticker: str, year: int, min_threshold: float) -> float:
    """Compute volatility from cached prices."""
    ticker_data = prices_df[prices_df['ticker'] == ticker]
    year_prices = ticker_data[ticker_data['date'].dt.year == year]['price']
    if len(year_prices) > 1:
        vol = year_prices.pct_change().std()
        return max(float(vol) if not pd.isna(vol) else min_threshold, min_threshold)
    return min_threshold

def update_volatility_cache(data_dir: str | Path) -> None:
    """Compute and cache volatility thresholds from prices cache."""
    import json
    data_dir = Path(data_dir)
    cache_path = data_dir / VOLATILITY_CACHE_FILENAME

    print(f"Loading raw data...")
    raw_df = load_raw_data(data_dir)

    print(f"Computing volatility thresholds from prices cache...")
    volatility_dict = get_yearly_volatility_thresholds(data_dir, raw_df)

    # Convert to JSON-serializable format
    volatility_json = {}
    for ticker, years in volatility_dict.items():
        volatility_json[ticker] = {str(year): float(thresh) for year, thresh in years.items()}

    # Save to JSON
    with open(cache_path, 'w') as f:
        json.dump(volatility_json, f, indent=2)

    total_entries = sum(len(v) for v in volatility_json.values())
    print(f"✓ Saved {total_entries} volatility thresholds to {cache_path}")
    print(f"  {len(volatility_json)} tickers, spanning multiple years")

def update_prices_cache(df: pd.DataFrame, data_dir: str | Path, batch_size: int = 15) -> None:
    """Fetch historical prices from yfinance and cache them. Run once, then reuse."""
    cache_dir = Path(data_dir)
    prices_cache_path = cache_dir / PRICES_CACHE_FILENAME

    # Load existing price cache if available
    cached_prices = pd.DataFrame()
    if prices_cache_path.exists():
        cached_prices = pd.read_parquet(prices_cache_path)
        cached_prices['date'] = pd.to_datetime(cached_prices['date'])
        cached_tickers = set(cached_prices['ticker'].unique())
    else:
        cached_tickers = set()

    # Find which tickers we need
    required_tickers = set()
    all_years = set()
    for _, row in df.iterrows():
        tickers = row.get("mentioned_companies", [])
        if not isinstance(tickers, list):
            continue
        try:
            year = int(pd.to_datetime(row['date_publish']).year)
            all_years.add(year)
            for t in tickers:
                required_tickers.add(t)
        except Exception:
            continue

    # Find which tickers are missing from cache
    missing_tickers = sorted(list(required_tickers - cached_tickers))

    if not missing_tickers:
        print("✓ Price cache is up-to-date (all tickers cached)")
        return

    try:
        import yfinance as yf
        min_year, max_year = min(all_years), max(all_years)

        print(f"Downloading prices for {len(missing_tickers)} tickers ({min_year}-{max_year})...")
        print(f"  Batching {batch_size} tickers per API call\n")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Batch download missing tickers
        all_new_prices = []
        for batch_idx in range(0, len(missing_tickers), batch_size):
            batch = missing_tickers[batch_idx : batch_idx + batch_size]
            new_prices = _fetch_and_cache_prices(batch, min_year, max_year, prices_cache_path)

            if not new_prices.empty:
                all_new_prices.append(new_prices)
                sample_rows = new_prices.groupby('ticker').size().head(2)
                print(f"  ✓ Batch {batch_idx // batch_size + 1}: {', '.join(f'{t}({n}d)' for t, n in sample_rows.items())}")
            else:
                print(f"  ⚠ Batch {batch_idx // batch_size + 1}: No data")

        # Combine with existing cache
        if all_new_prices:
            all_prices = pd.concat([cached_prices, *all_new_prices], ignore_index=True)
            all_prices.to_parquet(prices_cache_path, index=False)
            summary = all_prices.groupby('ticker')['date'].agg(['min', 'max', 'count'])
            print(f"\n✓ Prices cached: {prices_cache_path}")
            print(f"  {len(all_prices)} price points for {all_prices['ticker'].nunique()} tickers")
            print(f"  Date range: {summary['min'].min().date()} to {summary['max'].max().date()}")
    except ImportError:
        print("Error: yfinance not installed. Install with: pip install yfinance")

def calculate_label(row: pd.Series, ticker: str, threshold: float) -> int:
    curr = row.get(f"curr_day_price_{ticker}")
    nxt = row.get(f"next_day_price_{ticker}")
    if pd.isna(curr) or pd.isna(nxt) or curr == 0:
        return np.nan
    ret = (nxt - curr) / curr
    if ret > threshold: return 1
    if ret < -threshold: return -1
    return 0

def extract_aliases(ticker: str, named_entities: List[Dict], company_mapping: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    known_aliases = company_mapping.get(ticker, [])
    target_aliases = {ticker}
    if len(ticker) > 1:
        target_aliases.add(ticker.lower())

    for a in known_aliases:
        target_aliases.add(a)
        if len(a) > 1:
            target_aliases.add(a.lower())
        short = re.sub(r'\b(inc|corp|corporation|ltd|limited|group|plc)\b\.?', '', a, flags=re.IGNORECASE).strip()
        if len(short) > 1:
            target_aliases.add(short)
            target_aliases.add(short.lower())

    other_aliases = set()

    valid_patterns = [p for p in target_aliases if len(p) > 1]
    pattern_str = "|".join(map(re.escape, sorted(list(valid_patterns), key=len, reverse=True)))
    target_regex = re.compile(rf'(?<!\w)({pattern_str})(?!\w)', flags=re.IGNORECASE) if valid_patterns else None

    for ent in (named_entities or []):
        word = ent.get('word', '')
        if not word or len(word) < 2: continue

        is_target = (ent.get('company_key') == ticker) or (bool(target_regex.search(word)) if target_regex else False)

        if is_target:
            target_aliases.add(word)
            target_aliases.add(word.lower())
        elif ent.get('entity_group') == 'ORG':
            other_aliases.add(word)
            other_aliases.add(word.lower())

    return (sorted(list(target_aliases), key=len, reverse=True),
            sorted(list(other_aliases - target_aliases), key=len, reverse=True))

def segment_and_resolve_coreferences(text_or_doc) -> List[str]:
    """Segment text and resolve coreferences. Accepts either text string or spaCy doc (for caching)."""
    # If passed a spaCy doc, use it directly (cached)
    if hasattr(text_or_doc, 'sents'):
        doc = text_or_doc
    else:
        # Otherwise process text (fallback)
        text = text_or_doc
        if len(text) > 10000: text = text[:10000]
        if nlp is None: return [text]
        doc = nlp(text)

    resolved_sentences = []
    target_is_active = False

    for sent in doc.sents:
        sent_text = sent.text
        if TARGET_TOKEN in sent_text:
            target_is_active = True
            if OTHER_TOKEN in sent_text and sent_text.rfind(OTHER_TOKEN) > sent_text.rfind(TARGET_TOKEN):
                target_is_active = False
        elif OTHER_TOKEN in sent_text:
            target_is_active = False

        if target_is_active and not neural_coref_model:
            tokens = []
            for token in sent:
                if (token.pos_ == "PRON" and token.text.lower() in ["it", "its", "itself"]) or \
                   (token.text.lower() == "company" and token.i > 0 and doc[token.i-1].text.lower() == "the"):
                    if token.text.lower() == "company" and tokens and tokens[-1].strip().lower() == "the":
                        tokens.pop()
                    tokens.append(TARGET_TOKEN + token.whitespace_)
                else:
                    tokens.append(token.text_with_ws)
            resolved_sentences.append("".join(tokens).strip())
        else:
            resolved_sentences.append(sent_text.strip())

    return [s for s in resolved_sentences if s]

def _ent_mask(ent_type: str, ent_lemma: str, target_set: set, other_set: set) -> str | None:
    """Map a SpaCy entity type to a mask token. Returns None to skip masking."""
    if ent_type == 'ORG':
        return '[TARGET]' if ent_lemma in target_set else '[ENTITY]'
    if ent_type == 'PRODUCT':
        # Company-specific products (iPhone, iPad…) live in target_set
        return '[TARGET]' if ent_lemma in target_set else None
    if ent_type == 'PERSON':
        return '[TARGET]' if ent_lemma in target_set else '[PERSON]'
    if ent_type in ('DATE', 'TIME'):
        return '[DATE]'
    if ent_type == 'MONEY':
        return '[MONEY]'
    if ent_type == 'PERCENT':
        return '[PERCENT]'
    if ent_type in ('CARDINAL', 'ORDINAL', 'QUANTITY'):
        return '[NUMBER]'
    # GPE / LOC / NORP / EVENT / WORK_OF_ART … kept as plain text (geographic
    # context is meaningful in financial news)
    return None


def build_tfidf_track(text_or_doc, target_aliases: List[str], other_aliases: List[str]) -> str:
    """Build TF-IDF input using SpaCy NER + lemmatization.

    Masking strategy (priority order):
      - SpaCy NER entities  →  [TARGET] / [ENTITY] / [DATE] / [MONEY] / [PERCENT] / [NUMBER] / [PERSON]
      - Alias fallback (for NER misses, e.g. products)  →  [TARGET] / [ENTITY]
      - token.like_num fallback  →  [NUMBER]
    Stop words and punctuation are removed; remaining tokens are lemmatized.
    """
    if nlp is None:
        return text_or_doc.text if hasattr(text_or_doc, 'text') else str(text_or_doc)

    doc = text_or_doc if hasattr(text_or_doc, 'sents') else nlp(text_or_doc)

    # Pre-build lowercase lookup sets from aliases
    target_set = {a.lower() for a in target_aliases if len(a) > 1}
    other_set  = {a.lower() for a in other_aliases  if len(a) > 1}

    processed_sentences: List[str] = []
    target_found = False

    for sent in doc.sents:
        tokens_out: List[str] = []
        toks = list(sent)
        i = 0

        while i < len(toks):
            tok = toks[i]

            # ── Named entity span ────────────────────────────────────────────
            if tok.ent_iob_ == 'B':
                ent_type = tok.ent_type_
                span = [tok]
                i += 1
                while i < len(toks) and toks[i].ent_iob_ == 'I':
                    span.append(toks[i])
                    i += 1

                ent_lemma = ' '.join(t.lemma_ for t in span).lower()
                mask = _ent_mask(ent_type, ent_lemma, target_set, other_set)

                if mask is None:
                    # Not masked by NER — alias fallback then plain lemma
                    for t in span:
                        lm = t.lemma_.lower()
                        if not t.is_stop and not t.is_punct and lm.strip():
                            if lm in target_set:
                                tokens_out.append('[TARGET]')
                            elif lm in other_set:
                                tokens_out.append('[ENTITY]')
                            else:
                                tokens_out.append(lm)
                else:
                    tokens_out.append(mask)
                continue

            # ── Mid-entity continuation (already consumed above) ─────────────
            if tok.ent_iob_ == 'I':
                i += 1
                continue

            # ── Regular token (outside entity) ───────────────────────────────
            i += 1
            if tok.is_stop or tok.is_punct or tok.is_space or not tok.text.strip():
                continue

            lm = tok.lemma_.lower()

            # Alias fallback (NER may miss lowercased / inflected forms)
            if lm in target_set or tok.text.lower() in target_set:
                tokens_out.append('[TARGET]')
            elif lm in other_set or tok.text.lower() in other_set:
                tokens_out.append('[ENTITY]')
            elif tok.like_num or tok.is_currency:
                tokens_out.append('[NUMBER]')
            elif lm.strip():
                tokens_out.append(lm)

        if not tokens_out:
            continue

        # Collapse consecutive identical mask tokens
        deduped: List[str] = [tokens_out[0]]
        for t in tokens_out[1:]:
            if not (t.startswith('[') and t == deduped[-1]):
                deduped.append(t)

        sent_str = ' '.join(deduped).strip()
        if sent_str:
            if '[TARGET]' in sent_str:
                target_found = True
            processed_sentences.append(sent_str)

    # Filter: keep only sentences containing [TARGET] and one neighbour on each side
    if target_found:
        target_idx = [i for i, s in enumerate(processed_sentences) if '[TARGET]' in s]
        keep: set = set()
        for idx in target_idx:
            keep.update(range(max(0, idx - 1), min(len(processed_sentences), idx + 2)))
        processed_sentences = [processed_sentences[i] for i in sorted(keep)]

    return ' '.join(processed_sentences)

def build_bert_track(title: str, text_or_doc, target_aliases: List[str], other_aliases: List[str]) -> str:
    """Build BERT input. Accepts either text string or spaCy doc (for caching)."""
    # If passed a doc, extract text; otherwise use text directly
    if hasattr(text_or_doc, 'text'):
        text = text_or_doc.text
        doc = text_or_doc
    else:
        text = text_or_doc
        doc = None

    def mask(t, tgt_tag="[TARGET]", ent_tag="[ENTITY]"):
        if target_aliases:
            long_a = sorted([re.escape(a) for a in target_aliases if len(a) > 1], key=len, reverse=True)
            short_a = [re.escape(a) for a in target_aliases if len(a) == 1]
            if long_a:
                t = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', tgt_tag, str(t), flags=re.IGNORECASE)
            if short_a:
                t = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', tgt_tag, str(t))

        if other_aliases:
            long_a = sorted([re.escape(a) for a in other_aliases if len(a) > 1], key=len, reverse=True)
            short_a = [re.escape(a) for a in other_aliases if len(a) == 1]
            if long_a:
                t = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', ent_tag, str(t), flags=re.IGNORECASE)
            if short_a:
                t = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', ent_tag, str(t))

        return t

    m_title = mask(title)
    m_text = mask(text)

    m_title = re.sub(r'\[TARGET\](\s*[\(\[\s\-]*\[TARGET\][\)\]\s]*)+', '[TARGET]', m_title, flags=re.IGNORECASE)
    m_title = re.sub(r'\[ENTITY\](\s*[\(\[\s\-]*\[ENTITY\][\)\]\s]*)+', '[ENTITY]', m_title, flags=re.IGNORECASE)
    m_text = re.sub(r'\[TARGET\](\s*[\(\[\s\-]*\[TARGET\][\)\]\s]*)+', '[TARGET]', m_text, flags=re.IGNORECASE)
    m_text = re.sub(r'\[ENTITY\](\s*[\(\[\s\-]*\[ENTITY\][\)\]\s]*)+', '[ENTITY]', m_text, flags=re.IGNORECASE)

    if nlp is None:
        extracted = m_text
    else:
        # Use cached doc for sentence splitting if available, otherwise parse masked text
        if doc is not None:
            # Use original doc sentences, then apply masking
            sentences = [mask(sent.text.strip()) for sent in doc.sents]
        else:
            # Fallback: parse and mask
            parsed_doc = nlp(m_text)
            sentences = [sent.text.strip() for sent in parsed_doc.sents]

        target_indices = [i for i, s in enumerate(sentences) if "[TARGET]" in s]

        if target_indices and finbert_tokenizer:
            win = set(target_indices)
            radius = 1
            while len(finbert_tokenizer.tokenize(" ".join([sentences[i] for i in sorted(list(win))]))) < 400:
                added = False
                for idx in target_indices:
                    for side in [idx-radius, idx+radius]:
                        if 0 <= side < len(sentences) and side not in win:
                            win.add(side); added = True
                if not added: break
                radius += 1
            extracted = " ".join([sentences[i] for i in sorted(list(win))])
        else:
            extracted = m_text

    return f"[CLS] {m_title} [SEP] {extracted} [SEP]"

def saliency_filter(text_or_doc, title: str, aliases: List[str]) -> bool:
    """Check if target company is salient. Accepts either text string or spaCy doc (for caching)."""
    title_lower = str(title).lower()
    if any(alias.lower() in title_lower for alias in aliases if len(alias) > 1 or (len(alias)==1 and alias in str(title))):
        return True

    # If passed a spaCy doc, use it directly (cached)
    if hasattr(text_or_doc, 'sents'):
        doc = text_or_doc
    else:
        # Otherwise process text (fallback)
        temp_text = str(text_or_doc)
        if aliases:
            long_a = list(map(re.escape, filter(lambda a: len(a) > 1, aliases)))
            short_a = list(map(re.escape, filter(lambda a: len(a) == 1, aliases)))
            if long_a: temp_text = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', TARGET_TOKEN, temp_text, flags=re.IGNORECASE)
            if short_a: temp_text = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', TARGET_TOKEN, temp_text)
        if len(temp_text) > 10000: temp_text = temp_text[:10000]
        if nlp is None: return True
        doc = nlp(temp_text)

    score = 0
    for token in doc:
        if token.text == TARGET_TOKEN:
            score += 3 if token.dep_ in ['nsubj', 'nsubjpass', 'csubj'] else (2 if token.dep_ in ['dobj', 'pobj'] else 1)
            if score >= 3: return True
    return score >= 3

def _process_article(article_data: Dict) -> List[Dict]:
    """Process a single article for all its tickers. Called by multiprocessing.
    Uses cached SpaCy doc to avoid redundant NLP calls."""
    row, article_id, year, volatility_thresholds, company_mapping = article_data
    tickers = row.get("mentioned_companies", [])
    if not isinstance(tickers, list) or not tickers:
        return []

    # Neural coref once per article
    text = apply_neural_coref(row['maintext'])

    # Parse with SpaCy once, cache the doc
    if nlp is not None:
        doc = nlp(text)
    else:
        doc = None

    processed_rows = []

    for ticker in tickers:
        thresh = volatility_thresholds.get(ticker, {}).get(year, DEFAULT_THRESHOLD)
        label = calculate_label(row, ticker, thresh)
        if pd.isna(label):
            continue

        t_alias, o_alias = extract_aliases(ticker, row.get('named_entities', []), company_mapping)

        # EARLY EXIT: Check saliency BEFORE expensive BERT/TFIDF building
        # Pass cached doc if available
        if not saliency_filter(doc if doc is not None else text, row['title'], t_alias):
            continue

        # Only build expensive features if passes saliency filter
        curr_price = row.get(f"curr_day_price_{ticker}")
        next_price = row.get(f"next_day_price_{ticker}")
        ret = (next_price - curr_price) / curr_price if curr_price and next_price else np.nan

        processed_rows.append({
            "article_id": str(article_id),
            "date": pd.to_datetime(row['date_publish']),
            "ticker": ticker,
            "label": int(label),
            "bert_input": build_bert_track(row['title'], doc if doc is not None else text, t_alias, o_alias),
            "tfidf_input": build_tfidf_track(doc if doc is not None else text, t_alias, o_alias),
            "curr_day_price": float(curr_price) if curr_price else np.nan,
            "next_day_price": float(next_price) if next_price else np.nan,
            "return": ret,
            "volatility_threshold": thresh
        })

    return processed_rows


def process_pipeline(df: pd.DataFrame, data_dir: str | Path, mapping_path: str | Path = MAPPING_FILENAME, num_workers: int = 4, checkpoint_every: int = 10000) -> pd.DataFrame:
    """Process articles in parallel with checkpointing for resume capability."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # Print processing info
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    print(f"Processing {len(df)} articles ({num_workers} workers)")
    print(f"  GPU: {gpu_name}")
    print(f"  Neural coref: {neural_coref_device.upper() if neural_coref_model else 'DISABLED'}")
    print(f"  SpaCy: CPU (cached docs)")
    print(f"  Checkpoints: Every {checkpoint_every} articles\n")

    # Resolve mapping path: check current dir first, then data_dir, then as-is
    mapping_path_obj = Path(mapping_path)
    if not mapping_path_obj.is_absolute():
        if Path(mapping_path).exists():
            # File exists in current directory
            pass
        else:
            # Try in data_dir
            alt_path = Path(data_dir) / mapping_path
            if alt_path.exists():
                mapping_path = alt_path

    company_mapping = load_company_mapping(mapping_path)
    volatility_thresholds = get_yearly_volatility_thresholds(data_dir, df)
    article_id_counter = {}
    checkpoint_path = Path(data_dir) / ".processing_checkpoint.parquet"

    # Load existing checkpoint
    all_rows = []
    start_idx = 0
    processed_article_ids = set()
    if checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_path)
            all_rows = checkpoint_df.to_dict('records')
            start_idx = len(checkpoint_df)
            processed_article_ids = set(checkpoint_df['article_id'].unique())
            print(f"✓ Resuming from checkpoint: {start_idx} samples already processed\n")
        except Exception:
            pass

    # Prepare article data for processing (skip already processed articles)
    articles_to_process = []
    for idx, (_, row) in enumerate(df.iterrows()):
        tickers = row.get("mentioned_companies", [])
        if not isinstance(tickers, list) or not tickers:
            continue

        orig_id = row.get('id')
        if orig_id is None:
            art_key = (str(row.get('title', '')), str(row.get('date_publish', '')))
            if art_key not in article_id_counter:
                article_id_counter[art_key] = len(article_id_counter) + 1
            orig_id = f"article_{article_id_counter[art_key]}"

        # Skip if already processed (deduplication)
        if orig_id in processed_article_ids:
            continue

        year = str(pd.to_datetime(row['date_publish']).year)
        articles_to_process.append((row, orig_id, year, volatility_thresholds, company_mapping))

    # Process articles in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_article, article) for article in articles_to_process]
        total_futures = len(futures)

        # Use tqdm if available, otherwise manual progress
        if tqdm:
            progress_bar = tqdm(total=total_futures, desc="Processing", unit="article", initial=start_idx)
        else:
            progress_bar = None

        completed = 0
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            completed += 1

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"samples": len(all_rows)})

            # Checkpoint every N articles
            if completed % checkpoint_every == 0:
                checkpoint_df = pd.DataFrame(all_rows)
                checkpoint_df.to_parquet(checkpoint_path, index=False)
                if not progress_bar:
                    print(f"  ✓ Checkpoint: {len(all_rows)} samples saved")

        if progress_bar:
            progress_bar.close()

    # Final save and cleanup
    result_df = pd.DataFrame(all_rows).sort_values("date") if all_rows else pd.DataFrame()
    result_df = result_df.reset_index(drop=True)

    # Remove checkpoint file on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return result_df

def get_processed_data(data_dir: str | Path, force_refresh: bool = False, mapping_path: str | Path = MAPPING_FILENAME, checkpoint_every: int = 10000) -> pd.DataFrame:
    """Load or process dataset with automatic checkpointing + resume.

    Args:
        force_refresh: If False, use cached parquet if exists. If True, reprocess (but resume from checkpoint).
        checkpoint_every: Save checkpoint every N articles for resume capability.

    Behavior:
        - force_refresh=False: Returns cached parquet if exists (instant)
        - force_refresh=True: Reprocesses all articles, but resumes from .processing_checkpoint.parquet if it exists
        - Checkpoint is auto-deleted on successful completion
        - Final result saved to processed_dataset.parquet
    """
    # Resolve mapping path: check current dir first, then data_dir, then as-is
    mapping_path_obj = Path(mapping_path)
    if not mapping_path_obj.is_absolute():
        if Path(mapping_path).exists():
            # File exists in current directory
            pass
        else:
            # Try in data_dir
            alt_path = Path(data_dir) / mapping_path
            if alt_path.exists():
                mapping_path = alt_path

    path = Path(data_dir) / CACHE_FILENAME
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    print(f"Processing data (checkpoints every {checkpoint_every} articles)...\n")

    raw_df = load_raw_data(data_dir)
    processed_df = process_pipeline(raw_df, data_dir, mapping_path, checkpoint_every=checkpoint_every)
    processed_df.to_parquet(path, index=False)
    print(f"\n✓ Dataset saved: {path}")
    return processed_df

def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = df[['article_id', 'date']].drop_duplicates().sort_values('date')
    tr, va = int(len(u) * 0.7), int(len(u) * 0.85)
    train_ids = set(u.iloc[:tr]['article_id'])
    val_ids = set(u.iloc[tr:va]['article_id'])
    test_ids = set(u.iloc[va:]['article_id'])

    return (df[df['article_id'].isin(train_ids)].copy(),
            df[df['article_id'].isin(val_ids)].copy(),
            df[df['article_id'].isin(test_ids)].copy())
