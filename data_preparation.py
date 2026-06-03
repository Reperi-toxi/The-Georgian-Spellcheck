"""
Georgian Spellcheck — Data Preparation
=======================================
Produces:
  data/georgian_spellcheck_dataset.csv   — (input_text, target_text) pairs
  data/char_vocab.json                   — character vocabulary + special token indices
  data/georgian_vocabulary.txt           — clean word list
  data/dataset_stats.png                 — composition / length charts

Usage:
  python data_preparation.py

No train/val split is done here. Handle that in your training notebook:
  from sklearn.model_selection import train_test_split
  train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
"""

import bz2
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
WIKI_DUMP_PATH   = Path(r"C:\Users\gigic\Downloads\kawiki-20251101-pages-articles-multistream.xml.bz2")
OUTPUT_DIR       = Path("data")

MIN_WORD_FREQ    = 3       # discard words seen fewer times than this
MIN_WORD_LEN     = 3       # discard shorter words
MAX_WORD_LEN     = 20      # discard longer compounds / runons
RANDOM_SEED      = 42

# Error generation
BASE_VARIANTS    = 3       # corrupted variants for a word at median frequency
MAX_EXTRA        = 3       # additional variants for very frequent words (freq-weighted)
IDENTITY_RATIO   = 0.25    # fraction of final pairs that are (correct → correct)
SINGLE_EDIT_PROB = 0.75    # probability of applying only 1 edit (rest get 2 edits)

# ── Georgian keyboard adjacency (Mkhedruli QWERTY layout) ─────────────────────
# Based on the physical key positions on a standard Georgian keyboard.
KEYBOARD_NEIGHBORS: dict[str, list[str]] = {
    'ქ': ['წ', 'ა', 'ს', 'ჭ'],
    'წ': ['ქ', 'ე', 'ს', 'დ', 'ა', 'ჭ'],
    'ჭ': ['ქ', 'ე', 'ს', 'დ', 'ა', 'წ'],
    'ე': ['წ', 'რ', 'დ', 'ფ', 'ს'],
    'რ': ['ე', 'ტ', 'ფ', 'გ', 'დ', 'ღ'],
    'ღ': ['ე', 'ტ', 'ფ', 'გ', 'დ', 'რ'],
    'ტ': ['რ', 'ყ', 'გ', 'ჰ', 'ფ', 'თ'],
    'თ': ['რ', 'ყ', 'გ', 'ჰ', 'ფ', 'ტ'],
    'ყ': ['ტ', 'უ', 'ჰ', 'ჯ', 'გ'],
    'უ': ['ყ', 'ი', 'ჯ', 'კ'],
    'ი': ['უ', 'ო', 'ჯ', 'ლ', 'კ'],
    'ო': ['ი', 'პ', 'ლ', 'კ'],
    'პ': ['ო', 'ლ'],
    'ა': ['ქ', 'ს', 'ზ', 'წ'],
    'ს': ['ა', 'დ', 'ზ', 'ხ', 'წ', 'ქ', 'ე', 'შ'],
    'შ': ['ა', 'დ', 'ზ', 'ხ', 'ს', 'ე'],
    'დ': ['ს', 'ფ', 'ხ', 'ც', 'ე'],
    'ფ': ['დ', 'გ', 'ც', 'ვ', 'რ', 'ტ'],
    'გ': ['ფ', 'ჰ', 'ვ', 'ბ', 'ტ', 'ყ'],
    'ჰ': ['გ', 'ჯ', 'ბ', 'ნ', 'ყ', 'უ'],
    'ჯ': ['ჰ', 'კ', 'ნ', 'მ', 'უ', 'ი', 'ჟ'],
    'ჟ': ['ჰ', 'კ', 'ნ', 'მ', 'უ', 'ი', 'ჯ'],
    'კ': ['ჯ', 'ი', 'ო', 'ლ', 'მ'],
    'ლ': ['კ', 'ო', 'პ', 'მ'],
    'ზ': ['ა', 'ს', 'ხ', 'ძ'],
    'ძ': ['ა', 'ს', 'ხ', 'ზ'],
    'ხ': ['ზ', 'დ', 'ც', 'ს'],
    'ც': ['ხ', 'ფ', 'ვ', 'დ', 'ჩ'],
    'ჩ': ['ხ', 'ფ', 'ვ', 'დ', 'ც'],
    'ვ': ['ც', 'გ', 'ბ', 'ფ'],
    'ბ': ['ვ', 'ჰ', 'ნ', 'გ'],
    'ნ': ['ბ', 'ჯ', 'მ', 'ჰ'],
    'მ': ['ნ', 'ჯ', 'კ', 'ლ'],
}

# ── Phonetic confusion pairs (ejective vs. aspirate / voiced) ─────────────────
# These are the most linguistically realistic errors in Georgian:
# writers confuse phonetically similar consonant pairs.
PHONETIC_PAIRS: dict[str, str] = {
    'კ': 'ქ', 'ქ': 'კ',   # voiceless stop — ejective vs. aspirate
    'ტ': 'თ', 'თ': 'ტ',   # dental stop
    'პ': 'ფ', 'ფ': 'პ',   # labial stop
    'ც': 'წ', 'წ': 'ც',   # affricate
    'ჩ': 'ჭ', 'ჭ': 'ჩ',   # palatal affricate
    'ძ': 'ზ', 'ზ': 'ძ',   # voiced affricate vs. fricative
    'ღ': 'გ', 'გ': 'ღ',   # velar fricative vs. stop
    'ხ': 'ჰ', 'ჰ': 'ხ',   # fricatives
    'შ': 'ს', 'ს': 'შ',   # sibilants
    'ჟ': 'ზ',              # voiced sibilants
    'რ': 'ლ', 'ლ': 'რ',   # liquids
}

GEORGIAN_ALPHABET = list('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')
GEORGIAN_WORD_RE  = re.compile(r'^[\u10D0-\u10FA]+$')

# Error type weights — must sum to 1.0
_ERROR_TYPE_WEIGHTS = [
    ('keyboard_swap', 0.35),
    ('transposition', 0.25),
    ('omission',      0.20),
    ('phonetic_swap', 0.12),
    ('insertion',     0.08),
]
_ERROR_TYPES, _ERROR_PROBS = zip(*_ERROR_TYPE_WEIGHTS)


# ── MediaWiki markup strippers ─────────────────────────────────────────────────
_RE_TEMPLATE   = re.compile(r'\{\{.*?\}\}', re.DOTALL)
_RE_LINK       = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')
_RE_EXTERNAL   = re.compile(r'\[https?://\S+\s*([^\]]*)\]')
_RE_TAG        = re.compile(r'<[^>]+>')
_RE_HEADING    = re.compile(r'={2,}.*?={2,}')
_RE_BOLD_ITAL  = re.compile(r"'{2,}")
_RE_TABLE_CELL = re.compile(r'\|[^\n]+')
_RE_LIST_MARK  = re.compile(r'^[*#:;]+', re.MULTILINE)
_RE_HTML_ENT   = re.compile(r'&\w+;')


def _clean_wikitext(raw: str) -> str:
    text = _RE_TEMPLATE.sub(' ', raw)
    text = _RE_LINK.sub(r'\1', text)
    text = _RE_EXTERNAL.sub(r'\1', text)
    text = _RE_TAG.sub(' ', text)
    text = _RE_HEADING.sub(' ', text)
    text = _RE_BOLD_ITAL.sub(' ', text)
    text = _RE_TABLE_CELL.sub(' ', text)
    text = _RE_LIST_MARK.sub(' ', text)
    text = _RE_HTML_ENT.sub(' ', text)
    return text


# ── Step 1: Parse Wikipedia dump ──────────────────────────────────────────────

def build_word_frequency(dump_path: Path) -> Counter:
    """
    Stream-parse the bz2 Wikipedia XML dump and return a Counter of
    Georgian word frequencies.  Memory-safe: elem.clear() is called
    after each article so only one article lives in RAM at a time.
    """
    word_freq: Counter = Counter()
    article_count = 0
    # Try both namespace versions present in different dump vintages
    ns_candidates = [
        'http://www.mediawiki.org/xml/export-0.11/',
        'http://www.mediawiki.org/xml/export-0.10/',
    ]
    detected_ns = None

    print("Scanning Wikipedia dump — this takes a few minutes…")

    with bz2.open(dump_path, 'rb') as fh:
        for event, elem in ET.iterparse(fh, events=('start', 'end')):

            # Auto-detect namespace from the first root tag
            if detected_ns is None and event == 'start':
                tag = elem.tag
                for ns in ns_candidates:
                    if tag == f'{{{ns}}}mediawiki':
                        detected_ns = ns
                        break
                if detected_ns is None:
                    # Extract namespace from whatever root tag we see
                    m = re.match(r'\{(.+?)\}', tag)
                    detected_ns = m.group(1) if m else ns_candidates[0]

            if event != 'end':
                continue

            text_tag = f'{{{detected_ns}}}text'
            if elem.tag != text_tag or not elem.text:
                elem.clear()
                continue

            raw = elem.text

            # Skip redirects
            if raw.lstrip().startswith(('#REDIRECT', '#გადამისამართება')):
                elem.clear()
                continue

            cleaned = _clean_wikitext(raw)

            for token in cleaned.split():
                # Strip common punctuation that survives markup removal
                word = token.strip('.,!?"()[]{}«»:-—;""„…|=*#\n\t')
                word_lower = word.lower()
                if (GEORGIAN_WORD_RE.match(word_lower)
                        and MIN_WORD_LEN <= len(word_lower) <= MAX_WORD_LEN):
                    word_freq[word_lower] += 1

            elem.clear()
            article_count += 1

            if article_count % 10_000 == 0:
                print(f"  {article_count:>8,} articles | "
                      f"{len(word_freq):>8,} unique words so far")

    print(f"\nDone: {article_count:,} articles scanned")
    return word_freq


# ── Step 2: Filter vocabulary ─────────────────────────────────────────────────

def filter_vocabulary(word_freq: Counter) -> list[str]:
    """
    Keep only words that:
      - appear at least MIN_WORD_FREQ times (removes hapaxes and OCR artifacts)
      - consist entirely of Georgian Mkhedruli characters
      - are within the configured length range
    Returns a list sorted by descending frequency.
    """
    vocab = [
        word for word, freq in word_freq.items()
        if freq >= MIN_WORD_FREQ
    ]
    # Sort by frequency descending so frequency-weighted logic is meaningful
    vocab.sort(key=lambda w: word_freq[w], reverse=True)
    print(f"Vocabulary after freq≥{MIN_WORD_FREQ} filter: {len(vocab):,} words "
          f"(removed {len(word_freq) - len(vocab):,} rare words)")
    return vocab


# ── Step 3: Error injection ───────────────────────────────────────────────────

def _apply_one_error(chars: list[str]) -> list[str]:
    """
    Apply a single random error to `chars` (in-place copy returned).
    Returns the same list unmodified if no valid error could be applied.
    """
    chars = chars[:]  # always work on a copy
    n = len(chars)
    if n < 3:
        return chars

    error_type = random.choices(_ERROR_TYPES, weights=_ERROR_PROBS, k=1)[0]
    idx = random.randint(0, n - 1)

    if error_type == 'omission':
        # Delete the character at idx
        chars.pop(idx)

    elif error_type == 'insertion':
        # Insert a random neighbor of the character at idx (or any Georgian
        # letter if that character has no defined neighbors).
        ch = chars[idx]
        pool = KEYBOARD_NEIGHBORS.get(ch) or GEORGIAN_ALPHABET
        chars.insert(idx, random.choice(pool))   # insert BEFORE idx, not after

    elif error_type == 'keyboard_swap':
        ch = chars[idx]
        neighbors = KEYBOARD_NEIGHBORS.get(ch)
        if neighbors:
            chars[idx] = random.choice(neighbors)
        # If no neighbors are defined, skip — do NOT substitute random garbage

    elif error_type == 'phonetic_swap':
        ch = chars[idx]
        if ch in PHONETIC_PAIRS:
            chars[idx] = PHONETIC_PAIRS[ch]
        # If no phonetic pair exists, skip cleanly

    elif error_type == 'transposition':
        # Swap idx with idx+1; clamp to avoid out-of-bounds
        if idx >= n - 1:
            idx = n - 2
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return chars


def corrupt(word: str, n_edits: int = 1) -> str:
    """
    Apply exactly `n_edits` errors to `word`.
    Guarantees the result differs from the input (retries up to 10 times).
    """
    for _ in range(10):
        chars = list(word)
        for _ in range(n_edits):
            chars = _apply_one_error(chars)
        result = ''.join(chars)
        if result != word and len(result) >= 2:
            return result
    # Fallback: guaranteed deletion if all attempts produced identity
    if len(word) > 2:
        idx = random.randint(1, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    return word


# ── Step 4: Build dataset ─────────────────────────────────────────────────────

def build_dataset(vocab: list[str], word_freq: Counter) -> pd.DataFrame:
    """
    For each word in `vocab`:
      - Generate frequency-weighted number of corrupted variants
      - Generate identity pair (correct → correct)

    Then pad with additional identity pairs so that IDENTITY_RATIO of the
    total dataset is identity pairs.
    """
    random.seed(RANDOM_SEED)

    max_freq = max(word_freq[w] for w in vocab)
    pairs: list[dict] = []

    print(f"Building dataset for {len(vocab):,} words…")

    for i, word in enumerate(vocab):
        if i % 50_000 == 0 and i > 0:
            print(f"  {i:>8,} / {len(vocab):,} words processed")

        # Frequency-weighted variant count (more frequent words get more variants)
        freq_ratio     = word_freq[word] / max_freq
        target_corrupt = BASE_VARIANTS + round(freq_ratio * MAX_EXTRA)

        # Max edit distance scales conservatively with word length:
        #   len 3–5  → max 1 edit
        #   len 6–9  → max 2 edits
        #   len 10+  → max 2 edits  (cap at 2; 3-edit examples hurt more than help)
        max_edits = 1 if len(word) < 6 else 2

        # Identity pair — always included once per word
        pairs.append({'input_text': word, 'target_text': word})

        # Corrupted pairs
        seen_corruptions: set[str] = set()
        attempts = 0
        generated = 0

        while generated < target_corrupt and attempts < target_corrupt * 8:
            attempts += 1
            n_edits  = 1 if random.random() < SINGLE_EDIT_PROB else min(2, max_edits)
            corrupted = corrupt(word, n_edits)
            if corrupted not in seen_corruptions:
                seen_corruptions.add(corrupted)
                pairs.append({'input_text': corrupted, 'target_text': word})
                generated += 1

    df = pd.DataFrame(pairs)
    df = df.drop_duplicates(subset=['input_text', 'target_text']).reset_index(drop=True)

    # ── Balance: add extra identity pairs to reach IDENTITY_RATIO ─────────
    n_error    = (df['input_text'] != df['target_text']).sum()
    n_identity_current = len(df) - n_error
    # Solve: n_identity_target / (n_error + n_identity_target) = IDENTITY_RATIO
    n_identity_target = int(n_error * IDENTITY_RATIO / (1 - IDENTITY_RATIO))
    n_extra = max(0, n_identity_target - n_identity_current)

    if n_extra > 0:
        extra_words = random.choices(vocab, k=n_extra)
        extra_pairs = [{'input_text': w, 'target_text': w} for w in extra_words]
        df = pd.concat([df, pd.DataFrame(extra_pairs)], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ── Step 5: Build character vocabulary ────────────────────────────────────────

def build_char_vocab(df: pd.DataFrame) -> dict:
    """
    Build char↔index mappings from all characters present in the dataset.
    Includes <PAD>, <SOS>, <EOS>, <UNK> special tokens at fixed indices 0–3.
    """
    all_chars: set[str] = set()
    for col in ('input_text', 'target_text'):
        for word in df[col]:
            all_chars.update(word)

    SPECIAL = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    char_list = SPECIAL + sorted(all_chars)
    char2idx  = {ch: i for i, ch in enumerate(char_list)}
    idx2char  = {i: ch for ch, i in char2idx.items()}

    return {
        'chars'      : char_list,
        'char2idx'   : char2idx,
        'idx2char'   : {str(k): v for k, v in idx2char.items()},
        'PAD_IDX'    : char2idx['<PAD>'],
        'SOS_IDX'    : char2idx['<SOS>'],
        'EOS_IDX'    : char2idx['<EOS>'],
        'UNK_IDX'    : char2idx['<UNK>'],
        'vocab_size' : len(char_list),
    }


# ── Step 6: Save outputs ──────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame,
                 vocab: list[str],
                 char_vocab: dict,
                 word_freq: Counter) -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Dataset CSV
    dataset_path = OUTPUT_DIR / 'georgian_spellcheck_dataset.csv'
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    size_mb = dataset_path.stat().st_size / 1e6
    print(f"Dataset saved  → {dataset_path}  ({size_mb:.1f} MB, {len(df):,} rows)")

    # Character vocabulary JSON
    char_vocab_path = OUTPUT_DIR / 'char_vocab.json'
    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)
    print(f"Char vocab saved → {char_vocab_path}  ({char_vocab['vocab_size']} tokens)")

    # Word list TXT
    vocab_path = OUTPUT_DIR / 'georgian_vocabulary.txt'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for w in sorted(vocab):
            f.write(f"{w}\n")
    print(f"Word list saved  → {vocab_path}  ({len(vocab):,} words)")

    # Stats plot
    n_error    = (df['input_text'] != df['target_text']).sum()
    n_identity = len(df) - n_error
    lengths    = df['target_text'].str.len()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(lengths, bins=range(MIN_WORD_LEN, MAX_WORD_LEN + 2),
                 color='steelblue', edgecolor='white')
    axes[0].set_title('Target word length distribution')
    axes[0].set_xlabel('Characters')
    axes[0].set_ylabel('Pairs')

    axes[1].bar(['Error pairs', 'Identity pairs'], [n_error, n_identity],
                color=['tomato', 'mediumseagreen'], edgecolor='white')
    axes[1].set_title('Dataset composition')
    axes[1].set_ylabel('Pairs')

    top_words  = [w for w, _ in word_freq.most_common(20)]
    top_counts = [word_freq[w] for w in top_words]
    axes[2].barh(top_words[::-1], top_counts[::-1], color='slateblue', edgecolor='white')
    axes[2].set_title('Top 20 most frequent words')
    axes[2].set_xlabel('Frequency in dump')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'dataset_stats.png'
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Plot saved       → {plot_path}")


# ── Step 7: Print summary ─────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, char_vocab: dict) -> None:
    n_error    = (df['input_text'] != df['target_text']).sum()
    n_identity = len(df) - n_error

    print()
    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total pairs          : {len(df):>10,}")
    print(f"  Error pairs          : {n_error:>10,}  ({n_error/len(df)*100:.1f}%)")
    print(f"  Identity pairs       : {n_identity:>10,}  ({n_identity/len(df)*100:.1f}%)")
    print(f"  Unique target words  : {df['target_text'].nunique():>10,}")
    print(f"  Character vocab size : {char_vocab['vocab_size']:>10,}")
    print(f"  Special token indices:")
    print(f"    <PAD> = {char_vocab['PAD_IDX']}")
    print(f"    <SOS> = {char_vocab['SOS_IDX']}")
    print(f"    <EOS> = {char_vocab['EOS_IDX']}")
    print(f"    <UNK> = {char_vocab['UNK_IDX']}")
    print("=" * 55)
    print()
    print("Sample pairs (10 random):")
    print(f"  {'INPUT':<22} {'TARGET':<22} {'TYPE'}")
    print("  " + "-" * 55)
    sample = df.sample(10, random_state=0)
    for _, row in sample.iterrows():
        pair_type = "identity" if row['input_text'] == row['target_text'] else "error"
        print(f"  {row['input_text']:<22} {row['target_text']:<22} {pair_type}")
    print()
    print("Load in your training notebook:")
    print("  import pandas as pd, json")
    print("  df = pd.read_csv('data/georgian_spellcheck_dataset.csv')")
    print("  char_vocab = json.load(open('data/char_vocab.json', encoding='utf-8'))")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    if not WIKI_DUMP_PATH.exists():
        print(f"ERROR: Wiki dump not found at {WIKI_DUMP_PATH}")
        print("Update WIKI_DUMP_PATH at the top of the script.")
        sys.exit(1)

    # 1. Parse dump → word frequencies
    word_freq = build_word_frequency(WIKI_DUMP_PATH)

    # 2. Filter vocabulary
    vocab = filter_vocabulary(word_freq)

    if len(vocab) < 1000:
        print(f"WARNING: vocabulary only has {len(vocab)} words. "
              f"Consider lowering MIN_WORD_FREQ (currently {MIN_WORD_FREQ}).")

    # 3. Build (input, target) pairs
    df = build_dataset(vocab, word_freq)

    # 4. Build character vocabulary
    char_vocab = build_char_vocab(df)

    # 5. Save everything
    save_outputs(df, vocab, char_vocab, word_freq)

    # 6. Print summary
    print_summary(df, char_vocab)