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
try:
    from fastcoref import FCoref
    import fastcoref.modeling
    
    # This attribute was added in newer transformers versions. 
    # If fastcoref's base class doesn't have it, we inject it manually to prevent the AttributeError.
    if hasattr(fastcoref.modeling, 'FCorefModel'):
        if not hasattr(fastcoref.modeling.FCorefModel, 'all_tied_weights_keys'):
            fastcoref.modeling.FCorefModel.all_tied_weights_keys = []
            
    # Check for AMD GPU (ROCm uses 'cuda' identifier)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_coref_model = FCoref(device=device) 
    print(f"Neural Coref loaded successfully on device: {device}")
except ImportError:
    warnings.warn("fastcoref not found. Run `uv pip install fastcoref`. Falling back to SpaCy heuristic coref.")
    neural_coref_model = None
except Exception as e:
    warnings.warn(f"Neural Coref failed to initialize (Patch failed): {e}")
    neural_coref_model = None

# --- 2. SpaCy & Tokenizer Integrations ---
import spacy
from transformers import AutoTokenizer

# Load spaCy for robust syntactic dependency parsing and sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
except OSError:
    warnings.warn("Downloading spaCy en_core_web_sm model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

# We use the FinBERT tokenizer to physically count tokens to prevent truncation
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
except Exception:
    warnings.warn("Transformers not found. Token-bound windows will fallback to sentence counts.")
    finbert_tokenizer = None

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
    return df

def get_yearly_volatility_thresholds(df: pd.DataFrame, data_dir: str | Path, min_threshold: float = 0.005, force_refresh: bool = False) -> Dict[str, Dict[str, float]]:
    import yfinance as yf
    cache_path = Path(data_dir) / VOLATILITY_CACHE_FILENAME
    cache = {}
    
    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
    if not yf:
        return cache

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
        print(f"Fetching historical volatility from yfinance for {len(missing_pairs)} (ticker, year) combinations...")
        missing_by_ticker = defaultdict(list)
        for t, y in missing_pairs:
            missing_by_ticker[t].append(int(y))
            
        for ticker, years in missing_by_ticker.items():
            if ticker not in cache:
                cache[ticker] = {}
            try:
                hist = yf.Ticker(ticker).history(start=f"{min(years)}-01-01", end=f"{max(years) + 1}-01-01")
                if not hist.empty:
                    hist['Return'] = hist['Close'].pct_change()
                    for year in years:
                        year_data = hist[hist.index.year == year]
                        if not year_data.empty:
                            std_dev = float(year_data['Return'].std())
                            cache[ticker][str(year)] = max(min_threshold, std_dev) if pd.notna(std_dev) else min_threshold
                        else:
                            cache[ticker][str(year)] = min_threshold
                else:
                    for year in years: cache[ticker][str(year)] = min_threshold
            except Exception:
                for year in years: cache[ticker][str(year)] = min_threshold
                    
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)
            
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
    target_aliases = {ticker, ticker.lower(), "entity", "target"}
    
    for a in known_aliases:
        target_aliases.add(a)
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
        text = re.sub(rf'(?<!\w)({"|".join(map(re.escape, target_aliases))})(?!\w)', TARGET_TOKEN, text, flags=re.IGNORECASE)
    # 2. Mask others
    if other_aliases:
        text = re.sub(rf'(?<!\w)({"|".join(map(re.escape, other_aliases))})(?!\w)', OTHER_TOKEN, text, flags=re.IGNORECASE)
    
    sentences = segment_and_resolve_coreferences(text)
    
    target_indices = [i for i, s in enumerate(sentences) if TARGET_TOKEN in s]
    valid_indices = set()
    for idx in target_indices:
        valid_indices.update(range(max(0, idx - 1), min(len(sentences), idx + 2)))

    processed_sentences = []
    for i in sorted(list(valid_indices or range(len(sentences)))):
        sentence = sentences[i]
        if TARGET_TOKEN in sentence:
            words = [f"tgt_{re.sub(r'[^\w\s]', '', w).lower()}" if re.sub(r'[^\w\s]', '', w).lower() not in [TARGET_TOKEN.lower(), OTHER_TOKEN.lower()] else w.lower() for w in sentence.split()]
            processed_sentences.append(" ".join(words))
        else:
            words = [re.sub(r'[^\w\s]', '', w).lower() for w in sentence.split() if w]
            processed_sentences.append(" ".join(words))
            
    return " ".join(processed_sentences)

def build_bert_track(title: str, text: str, target_aliases: List[str], other_aliases: List[str]) -> str:
    def mask(t, tgt_tag="[TARGET]", ent_tag="[ENTITY]"):
        if target_aliases:
            t = re.sub(rf'(?<!\w)({"|".join(map(re.escape, target_aliases))})(?!\w)', tgt_tag, str(t), flags=re.IGNORECASE)
        if other_aliases:
            t = re.sub(rf'(?<!\w)({"|".join(map(re.escape, other_aliases))})(?!\w)', ent_tag, str(t), flags=re.IGNORECASE)
        return t

    m_title = mask(title)
    m_text = mask(text)
    
    temp_text = m_text.replace("[TARGET]", TARGET_TOKEN).replace("[ENTITY]", OTHER_TOKEN)
    sentences = [s.replace(TARGET_TOKEN, "[TARGET]").replace(OTHER_TOKEN, "[ENTITY]") for s in segment_and_resolve_coreferences(temp_text)]

    target_indices = [i for i, s in enumerate(sentences) if "[TARGET]" in s]
    
    if target_indices and finbert_tokenizer:
        win = set(target_indices)
        radius = 1
        while len(finbert_tokenizer.tokenize(" ".join([sentences[i] for i in sorted(list(win))]))) < 450:
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

    return f"What is the financial sentiment impact for the entity [TARGET] in this article? [SEP] {m_title} - {extracted}"

def saliency_filter(text: str, title: str, aliases: List[str]) -> bool:
    title_lower = str(title).lower()
    if any(alias.lower() in title_lower for alias in aliases):
        return True
    
    temp_text = re.sub(rf'(?<!\w)({"|".join(map(re.escape, aliases))})(?!\w)', TARGET_TOKEN, str(text), flags=re.IGNORECASE) if aliases else str(text)
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
    
    for _, row in df.iterrows():
        tickers = row.get("mentioned_companies", [])
        if not isinstance(tickers, list) or not tickers: continue
        
        art_id = hash(str(row.get('title', '')) + str(row.get('date_publish', '')))
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
                "article_id": str(art_id),
                "date": pd.to_datetime(row['date_publish']),
                "ticker": ticker,
                "label": int(label),
                "bert_input": build_bert_track(row['title'], text, t_alias, o_alias),
                "tfidf_input": build_tfidf_track(text, t_alias, o_alias)
            })
            
    return pd.DataFrame(processed_rows).sort_values("date")

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