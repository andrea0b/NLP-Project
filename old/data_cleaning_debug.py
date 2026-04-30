import logging
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Suppress logging FIRST, before importing libraries
logging.basicConfig(level=logging.CRITICAL, force=True)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
logging.getLogger('fastcoref').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

import json
import lzma
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import pandas as pd
import numpy as np
import torch

# --- 1. THE MONKEY PATCH: Compatibility fix for fastcoref + new Transformers versions ---
neural_coref_model = None
try:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        from fastcoref import FCoref
        import fastcoref.modeling

        try:
            # Check for AMD GPU (ROCm uses 'cuda' identifier)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            neural_coref_model = FCoref(device=device)
        except Exception as patch_error:
            pass

except ImportError:
    pass

# --- 2. SpaCy & Tokenizer Integrations ---
with redirect_stdout(StringIO()):
    import spacy
    from transformers import AutoTokenizer

# Load spaCy for robust syntactic dependency parsing and sentence segmentation
try:
    with redirect_stdout(StringIO()):
        nlp = spacy.load("en_core_web_sm", disable=['ner'])
except OSError:
    try:
        with redirect_stdout(StringIO()):
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=['ner'])
    except Exception:
        nlp = None

# We use the FinBERT tokenizer to physically count tokens to prevent truncation
finbert_tokenizer = None
try:
    with redirect_stdout(StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
except Exception:
    pass

# ACL-Grade Constants
TARGET_TOKEN = "TGT_CO"
OTHER_TOKEN = "OTHER_CO"
DEFAULT_THRESHOLD = 0.005  
CACHE_FILENAME = "processed_dataset.parquet"
MAPPING_FILENAME = "company_mapping.json"
VOLATILITY_CACHE_FILENAME = "volatility_cache.json"

def apply_neural_coref(text: str) -> str:
    """
    SOTA: Replaces pronouns (it, its, they) with their explicit entity clusters 
    using a Neural Cross-Encoder prior to masking.
    """
    if not neural_coref_model or not isinstance(text, str) or not text.strip():
        return text
    try:
        preds = neural_coref_model.predict(texts=[text])
        return preds[0].get_resolved_text()
    except Exception as e:
        return text

def load_company_mapping(mapping_path: str | Path) -> Dict[str, List[str]]:
    path = Path(mapping_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Notice: Mapping file '{path}' not found. Defaulting to NER-only extraction.")
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
    df = df.head(5)  # DEBUG: Limit to 5 rows for faster testing
    return df

def get_yearly_volatility_thresholds(df: pd.DataFrame, data_dir: str | Path, min_threshold: float = 0.005, force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
    cache_path = Path(data_dir) / VOLATILITY_CACHE_FILENAME
    cache = {}

    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    required_pairs = set()
    for _, row in df.iterrows():
        tickers = row.get("mentioned_companies", [])
        if not isinstance(tickers, list): continue
        try:
            year = str(pd.to_datetime(row['date_publish']).year)
            for t in tickers:
                required_pairs.add((t, year))
        except Exception:
            continue

    missing_pairs = [(t, y) for t, y in required_pairs if t not in cache or y not in cache.get(t, {})]

    if missing_pairs:
        try:
            import yfinance as yf
            missing_by_ticker = defaultdict(list)
            for t, y in missing_pairs:
                missing_by_ticker[t].append(int(y))

            missing_tickers = list(missing_by_ticker.keys())

            # Batch download with progress suppression
            with redirect_stdout(StringIO()):
                for ticker in missing_tickers:
                    if ticker not in cache:
                        cache[ticker] = {}
                    try:
                        # Download data for the years we need
                        years_to_fetch = sorted(set(missing_by_ticker[ticker]))
                        for year in years_to_fetch:
                            start_date = f"{year}-01-01"
                            end_date = f"{year}-12-31"
                            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                            if not data.empty:
                                volatility = data['Adj Close'].pct_change().std()
                                cache[ticker][str(year)] = max(volatility, min_threshold)
                            else:
                                cache[ticker][str(year)] = min_threshold
                    except Exception:
                        # Fallback to default threshold if download fails
                        for year in missing_by_ticker[ticker]:
                            cache[ticker][str(year)] = min_threshold

            # Save updated cache
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4)
        except ImportError:
            # yfinance not available, use defaults
            for t, y in missing_pairs:
                if t not in cache:
                    cache[t] = {}
                cache[t][str(y)] = min_threshold

    return cache

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
    """
    Robust alias extraction: Handles lowercasing, placeholder detection, and common corporate suffix stripping.
    """
    # 1. Start with the ticker and its variants
    known_aliases = company_mapping.get(ticker, [])
    target_aliases = {ticker, "entity", "target"}
    if len(ticker) > 1:
        target_aliases.add(ticker.lower())
    
    for a in known_aliases:
        target_aliases.add(a)
        if len(a) > 1:
            target_aliases.add(a.lower())
        # Strip common suffixes: "Apple Inc." -> "Apple"
        short = re.sub(r'\b(inc|corp|corporation|ltd|limited|group|plc)\b\.?', '', a, flags=re.IGNORECASE).strip()
        if len(short) > 1:
            target_aliases.add(short)
            target_aliases.add(short.lower())

    other_aliases = set()
    
    # 2. Build lookahead regex for matching
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

def segment_and_resolve_coreferences(text: str) -> List[str]:
    """Fallback spaCy Coref + Sentence Segmentation"""
    if len(text) > 10000: text = text[:10000]
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

def build_tfidf_track(text: str, target_aliases: List[str], other_aliases: List[str]) -> str:
    # 1. Mask targets first
    if target_aliases:
        long_a = list(map(re.escape, filter(lambda a: len(a) > 1, target_aliases)))
        short_a = list(map(re.escape, filter(lambda a: len(a) == 1, target_aliases)))
        if long_a: text = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', TARGET_TOKEN, text, flags=re.IGNORECASE)
        if short_a: text = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', TARGET_TOKEN, text)
    # 2. Mask others
    if other_aliases:
        long_a = list(map(re.escape, filter(lambda a: len(a) > 1, other_aliases)))
        short_a = list(map(re.escape, filter(lambda a: len(a) == 1, other_aliases)))
        if long_a: text = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', OTHER_TOKEN, text, flags=re.IGNORECASE)
        if short_a: text = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', OTHER_TOKEN, text)
    
    sentences = segment_and_resolve_coreferences(text)
    
    target_indices = [i for i, s in enumerate(sentences) if TARGET_TOKEN in s]
    valid_indices = set()
    for idx in target_indices:
        valid_indices.update(range(max(0, idx - 1), min(len(sentences), idx + 2)))

    processed_sentences = []
    for i in sorted(list(valid_indices or range(len(sentences)))):
        sentence = sentences[i]
        
        # 1. Protect tokens before global lower/alphanumeric cleanup
        # Using a GUID-like string to avoid any collision with potential text
        sentence = sentence.replace(TARGET_TOKEN, "ABCDEFG_TARGET_HIDDEN")
        sentence = sentence.replace(OTHER_TOKEN, "ABCDEFG_OTHER_HIDDEN")
        
        # 2. Lowercase and strip punctuation
        sentence = re.sub(r'[^\w\s]', ' ', sentence).lower()
        
        # 3. Restore to target placeholders (Clean format for FinBERT)
        sentence = sentence.replace("abcdefg_target_hidden", "[TARGET]")
        sentence = sentence.replace("abcdefg_other_hidden", "[ENTITY]")

        # 4. Deduplicate consecutive target/entity references
        sentence = re.sub(r'\b(\[TARGET\]\s+)+\[TARGET\]\b', '[TARGET]', sentence)
        sentence = re.sub(r'\b(\[ENTITY\]\s+)+\[ENTITY\]\b', '[ENTITY]', sentence)
        
        processed_sentences.append(" ".join(sentence.split()))
            
    return " ".join(processed_sentences)

def build_bert_track(title: str, text: str, target_aliases: List[str], other_aliases: List[str]) -> str:
    def mask(t, tgt_tag="[TARGET]", ent_tag="[ENTITY]"):
        if target_aliases:
            long_a = sorted([re.escape(a) for a in target_aliases if len(a) > 1], key=len, reverse=True)
            short_a = [re.escape(a) for a in target_aliases if len(a) == 1]
            
            # Mask long ones case-insensitive
            if long_a:
                t = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', tgt_tag, str(t), flags=re.IGNORECASE)
            
            # Mask short ones case-sensitive
            if short_a:
                t = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', tgt_tag, str(t))
            
        if other_aliases:
            long_a = sorted([re.escape(a) for a in other_aliases if len(a) > 1], key=len, reverse=True)
            short_a = [re.escape(a) for a in other_aliases if len(a) == 1]
            
            # Mask long ones case-insensitive
            if long_a:
                t = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', ent_tag, str(t), flags=re.IGNORECASE)
            
            # Mask short ones case-sensitive
            if short_a:
                t = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', ent_tag, str(t))
            
        return t

    m_title = mask(title)
    m_text = mask(text)
    
    # Clean up redundant masks (e.g., "[TARGET] ([TARGET])" -> "[TARGET]")
    m_title = re.sub(r'\[TARGET\](\s*[\(\[\s\-]*\[TARGET\][\)\]\s]*)+', '[TARGET]', m_title, flags=re.IGNORECASE)
    m_title = re.sub(r'\[ENTITY\](\s*[\(\[\s\-]*\[ENTITY\][\)\]\s]*)+', '[ENTITY]', m_title, flags=re.IGNORECASE)

    m_text = re.sub(r'\[TARGET\](\s*[\(\[\s\-]*\[TARGET\][\)\]\s]*)+', '[TARGET]', m_text, flags=re.IGNORECASE)
    m_text = re.sub(r'\[ENTITY\](\s*[\(\[\s\-]*\[ENTITY\][\)\]\s]*)+', '[ENTITY]', m_text, flags=re.IGNORECASE)
    
    # Use simple sentence segmenter if fastcoref prediction isn't needed here
    # Since we already have the resolved coreferences in 'text', we just need to chunk it.
    doc = nlp(m_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    target_indices = [i for i, s in enumerate(sentences) if "[TARGET]" in s]
    
    if target_indices and finbert_tokenizer:
        win = set(target_indices)
        radius = 1
        # Target length is usually 512 for BERT, but we reserve space for prompt/title
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

def saliency_filter(text: str, title: str, aliases: List[str]) -> bool:
    title_lower = str(title).lower()
    if any(alias.lower() in title_lower for alias in aliases if len(alias) > 1 or (len(alias)==1 and alias in str(title))):
        return True
    
    # Mask using case-insensitivity only for > 1 len aliases
    temp_text = str(text)
    if aliases:
        long_a = list(map(re.escape, filter(lambda a: len(a) > 1, aliases)))
        short_a = list(map(re.escape, filter(lambda a: len(a) == 1, aliases)))
        if long_a: temp_text = re.sub(rf'(?<!\w)({"|".join(long_a)})(?!\w)', TARGET_TOKEN, temp_text, flags=re.IGNORECASE)
        if short_a: temp_text = re.sub(rf'(?<!\w)({"|".join(short_a)})(?!\w)', TARGET_TOKEN, temp_text)

    if len(temp_text) > 10000: temp_text = temp_text[:10000]
    doc = nlp(temp_text)
    score = 0
    for token in doc:
        if token.text == TARGET_TOKEN:
            score += 3 if token.dep_ in ['nsubj', 'nsubjpass', 'csubj'] else (2 if token.dep_ in ['dobj', 'pobj'] else 1)
            if score >= 3: return True
    return score >= 3

def process_pipeline(df: pd.DataFrame, data_dir: str | Path, mapping_path: str | Path = MAPPING_FILENAME, force_refresh: bool = False) -> pd.DataFrame:
    processed_rows = []
    company_mapping = load_company_mapping(mapping_path)
    volatility_thresholds = get_yearly_volatility_thresholds(df, data_dir, force_refresh=force_refresh)
    article_id_counter = {}  # Track unique articles

    for _, row in df.iterrows():
        tickers = row.get("mentioned_companies", [])
        if not isinstance(tickers, list) or not tickers: continue

        # Use original id if available, otherwise create a clean sequential one
        orig_id = row.get('id')
        if orig_id is None:
            art_key = (str(row.get('title', '')), str(row.get('date_publish', '')))
            if art_key not in article_id_counter:
                article_id_counter[art_key] = len(article_id_counter) + 1
            orig_id = f"article_{article_id_counter[art_key]}"

        text = apply_neural_coref(row['maintext'])
        year = str(pd.to_datetime(row['date_publish']).year)

        for ticker in tickers:
            thresh = volatility_thresholds.get(ticker, {}).get(year, DEFAULT_THRESHOLD)
            label = calculate_label(row, ticker, thresh)
            if pd.isna(label): continue

            t_alias, o_alias = extract_aliases(ticker, row.get('named_entities', []), company_mapping)

            if not saliency_filter(text, row['title'], t_alias):
                continue

            processed_rows.append({
                "article_id": str(orig_id),
                "date": pd.to_datetime(row['date_publish']),
                "ticker": ticker,
                "label": int(label),
                "bert_input": build_bert_track(row['title'], text, t_alias, o_alias),
                "tfidf_input": build_tfidf_track(text, t_alias, o_alias)
            })

    result_df = pd.DataFrame(processed_rows).sort_values("date")
    result_df = result_df.reset_index(drop=True)  # Clean index
    return result_df

def get_processed_data(data_dir: str | Path, force_refresh: bool = False, mapping_path: str | Path = MAPPING_FILENAME) -> pd.DataFrame:
    path = Path(data_dir) / CACHE_FILENAME
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    
    raw_df = load_raw_data(data_dir)
    processed_df = process_pipeline(raw_df, data_dir, mapping_path, force_refresh)
    processed_df.to_parquet(path, index=False)
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