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

No train/val split is done here. Handle that in your training notebook.
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
WIKI_DUMP_PATH = Path(r"C:\Users\gigic\Downloads\kawiki-20251101-pages-articles-multistream.xml.bz2")
OUTPUT_DIR     = Path("data")

MIN_WORD_FREQ  = 3
MIN_WORD_LEN   = 3
MAX_WORD_LEN   = 20
RANDOM_SEED    = 42

# ── Error generation ───────────────────────────────────────────────────────────
# FIX 1: MAX_EXTRA reduced from 3 → 1.
#   Previously the most-frequent words got up to BASE_VARIANTS+3 = 6 variants.
#   With only ~30 possible single-edit corruptions per word, exhausting that
#   budget forced the retry loop to fall back to 2-edit corruptions, producing
#   garbage pairs like: პრივილეგიებჯსგწნ → პრივილეგიებისგან (4+ edits apart).
#   Capping at BASE_VARIANTS+1 = 4 keeps well within the single-edit budget.
BASE_VARIANTS  = 3
MAX_EXTRA      = 1     # was 3
IDENTITY_RATIO = 0.25

# ── Georgian keyboard adjacency (Mkhedruli QWERTY layout) ─────────────────────
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

PHONETIC_PAIRS: dict[str, str] = {
    'კ': 'ქ', 'ქ': 'კ',
    'ტ': 'თ', 'თ': 'ტ',
    'პ': 'ფ', 'ფ': 'პ',
    'ც': 'წ', 'წ': 'ც',
    'ჩ': 'ჭ', 'ჭ': 'ჩ',
    'ძ': 'ზ', 'ზ': 'ძ',
    'ღ': 'გ', 'გ': 'ღ',
    'ხ': 'ჰ', 'ჰ': 'ხ',
    'შ': 'ს', 'ს': 'შ',
    'ჟ': 'ზ',
    'რ': 'ლ', 'ლ': 'რ',
}

GEORGIAN_ALPHABET = list('აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ')
GEORGIAN_WORD_RE  = re.compile(r'^[\u10D0-\u10FA]+$')

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
    word_freq: Counter = Counter()
    article_count = 0
    ns_candidates = [
        'http://www.mediawiki.org/xml/export-0.11/',
        'http://www.mediawiki.org/xml/export-0.10/',
    ]
    detected_ns = None

    print("Scanning Wikipedia dump…")

    with bz2.open(dump_path, 'rb') as fh:
        for event, elem in ET.iterparse(fh, events=('start', 'end')):
            if detected_ns is None and event == 'start':
                tag = elem.tag
                for ns in ns_candidates:
                    if tag == f'{{{ns}}}mediawiki':
                        detected_ns = ns
                        break
                if detected_ns is None:
                    m = re.match(r'\{(.+?)\}', tag)
                    detected_ns = m.group(1) if m else ns_candidates[0]

            if event != 'end':
                continue

            text_tag = f'{{{detected_ns}}}text'
            if elem.tag != text_tag or not elem.text:
                elem.clear()
                continue

            raw = elem.text
            if raw.lstrip().startswith(('#REDIRECT', '#გადამისამართება')):
                elem.clear()
                continue

            cleaned = _clean_wikitext(raw)
            for token in cleaned.split():
                word = token.strip('.,!?"()[]{}«»:-—;""„…|=*#\n\t')
                word_lower = word.lower()
                if (GEORGIAN_WORD_RE.match(word_lower)
                        and MIN_WORD_LEN <= len(word_lower) <= MAX_WORD_LEN):
                    word_freq[word_lower] += 1

            elem.clear()
            article_count += 1
            if article_count % 10_000 == 0:
                print(f"  {article_count:>8,} articles | {len(word_freq):>8,} unique words")

    print(f"\nDone: {article_count:,} articles scanned")
    return word_freq


# ── Step 2: Filter vocabulary ─────────────────────────────────────────────────

def filter_vocabulary(word_freq: Counter) -> list[str]:
    vocab = [w for w, f in word_freq.items() if f >= MIN_WORD_FREQ]
    vocab.sort(key=lambda w: word_freq[w], reverse=True)
    print(f"Vocabulary after freq≥{MIN_WORD_FREQ} filter: {len(vocab):,} words "
          f"(removed {len(word_freq) - len(vocab):,} rare words)")
    return vocab


# ── Step 3: Error injection ───────────────────────────────────────────────────

def _apply_one_error(chars: list[str]) -> list[str]:
    """Apply exactly one random error. Returns a copy — never mutates input."""
    chars = chars[:]
    n = len(chars)
    if n < 3:
        return chars

    error_type = random.choices(_ERROR_TYPES, weights=_ERROR_PROBS, k=1)[0]
    idx = random.randint(0, n - 1)

    if error_type == 'omission':
        chars.pop(idx)

    elif error_type == 'insertion':
        ch   = chars[idx]
        pool = KEYBOARD_NEIGHBORS.get(ch) or GEORGIAN_ALPHABET
        chars.insert(idx, random.choice(pool))

    elif error_type == 'keyboard_swap':
        ch        = chars[idx]
        neighbors = KEYBOARD_NEIGHBORS.get(ch)
        if neighbors:
            chars[idx] = random.choice(neighbors)
        # No neighbors → skip cleanly (no random garbage substitution)

    elif error_type == 'phonetic_swap':
        ch = chars[idx]
        if ch in PHONETIC_PAIRS:
            chars[idx] = PHONETIC_PAIRS[ch]
        # No phonetic pair → skip cleanly

    elif error_type == 'transposition':
        if idx >= n - 1:
            idx = n - 2
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return chars


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance. Returns early if distance exceeds 2."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 99   # fast reject — guaranteed > MAX_EDIT_DIST
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb))
            # Early exit: entire row already exceeds 2
        if min(curr) > 2:
            return 99
        prev = curr
    return prev[lb]


def corrupt(word: str) -> str:
    # FIX 2: n_edits is always 1. The 2-edit path is removed entirely.
    #   Previously: n_edits = 1 if random.random() < 0.75 else 2
    #   The 25% two-edit path was the direct cause of garbage pairs.
    #   Multi-edit corruptions pushed the retry loop past the single-edit
    #   budget, producing pairs 3–4 edits apart that the model cannot learn from.
    for _ in range(10):
        chars  = _apply_one_error(list(word))
        result = ''.join(chars)
        if result != word and len(result) >= 2:
            return result
    # Fallback: guaranteed single omission
    if len(word) > 2:
        idx = random.randint(1, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    return word


# ── Step 4: Build dataset ─────────────────────────────────────────────────────

def build_dataset(vocab: list[str], word_freq: Counter) -> pd.DataFrame:
    random.seed(RANDOM_SEED)

    max_freq = max(word_freq[w] for w in vocab)
    pairs: list[dict] = []

    print(f"Building dataset for {len(vocab):,} words…")

    for i, word in enumerate(vocab):
        if i % 50_000 == 0 and i > 0:
            print(f"  {i:>8,} / {len(vocab):,} words processed")

        # Frequency-weighted variant count.
        # FIX 3: MAX_EXTRA = 1 (was 3) so the max is BASE_VARIANTS+1 = 4.
        #   With ~30 possible single-edit corruptions for an 8-char word,
        #   4 variants = 13% coverage — well within budget, no fallback needed.
        freq_ratio     = word_freq[word] / max_freq
        target_corrupt = BASE_VARIANTS + round(freq_ratio * MAX_EXTRA)

        # Identity pair — always one per word
        pairs.append({'input_text': word, 'target_text': word})

        seen: set[str] = set()
        attempts = generated = 0

        while generated < target_corrupt and attempts < target_corrupt * 8:
            attempts += 1
            corrupted = corrupt(word)

            # FIX 4: Hard edit-distance guard.
            #   Even though corrupt() now always calls _apply_one_error once,
            #   some error types (keyboard_swap, phonetic_swap) can silently
            #   skip when no valid substitution exists for a given character.
            #   In that edge case corrupt() returns the fallback omission, which
            #   is always edit-distance 1. But we guard explicitly here so that
            #   no pair with edit distance > 1 can ever enter the dataset,
            #   regardless of any future changes to the error functions.
            if corrupted in seen:
                continue
            if _edit_distance(corrupted, word) > 1:
                continue   # discard — should not happen, but belt-and-suspenders

            seen.add(corrupted)
            pairs.append({'input_text': corrupted, 'target_text': word})
            generated += 1

    df = pd.DataFrame(pairs)
    df = df.drop_duplicates(subset=['input_text', 'target_text']).reset_index(drop=True)

    # Balance: add extra identity pairs to reach IDENTITY_RATIO
    n_error           = (df['input_text'] != df['target_text']).sum()
    n_identity_target = int(n_error * IDENTITY_RATIO / (1 - IDENTITY_RATIO))
    n_extra           = max(0, n_identity_target - (len(df) - n_error))

    if n_extra > 0:
        extra = [{'input_text': w, 'target_text': w}
                 for w in random.choices(vocab, k=n_extra)]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ── Step 5: Build character vocabulary ────────────────────────────────────────

def build_char_vocab(df: pd.DataFrame) -> dict:
    all_chars: set[str] = set()
    for col in ('input_text', 'target_text'):
        for word in df[col]:
            all_chars.update(word)

    SPECIAL   = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    char_list = SPECIAL + sorted(all_chars)
    char2idx  = {ch: i for i, ch in enumerate(char_list)}
    idx2char  = {i: ch for ch, i in char2idx.items()}

    return {
        'chars'     : char_list,
        'char2idx'  : char2idx,
        'idx2char'  : {str(k): v for k, v in idx2char.items()},
        'PAD_IDX'   : char2idx['<PAD>'],
        'SOS_IDX'   : char2idx['<SOS>'],
        'EOS_IDX'   : char2idx['<EOS>'],
        'UNK_IDX'   : char2idx['<UNK>'],
        'vocab_size': len(char_list),
    }


# ── Step 6: Verify dataset quality ────────────────────────────────────────────

def verify_dataset(df: pd.DataFrame) -> None:
    """
    Sanity-check the dataset before saving.
    Prints edit-distance distribution for error pairs and fails loudly
    if any pair with edit distance > 1 is found.
    """
    print("\nVerifying dataset quality…")
    error_pairs = df[df['input_text'] != df['target_text']]

    dist_counts: Counter = Counter()
    violations: list[tuple] = []

    for _, row in error_pairs.sample(min(5000, len(error_pairs)),
                                     random_state=0).iterrows():
        d = _edit_distance(row['input_text'], row['target_text'])
        dist_counts[min(d, 5)] += 1
        if d > 1:
            violations.append((row['input_text'], row['target_text'], d))

    print("Edit-distance distribution (sample of 5,000 error pairs):")
    for dist in sorted(dist_counts):
        label = f"dist={dist}" if dist < 5 else "dist≥5"
        bar   = "█" * (dist_counts[dist] // 20)
        print(f"  {label}: {dist_counts[dist]:>5}  {bar}")

    if violations:
        print(f"\n  WARNING: {len(violations)} pairs with edit distance > 1 found in sample.")
        print("  Examples:")
        for inp, tgt, d in violations[:5]:
            print(f"    dist={d}  {inp} → {tgt}")
    else:
        print("  ✓ All sampled pairs have edit distance = 1")


# ── Step 7: Save outputs ──────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, vocab: list[str],
                 char_vocab: dict, word_freq: Counter) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset_path = OUTPUT_DIR / 'georgian_spellcheck_dataset.csv'
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    print(f"Dataset saved    → {dataset_path}  "
          f"({dataset_path.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")

    char_vocab_path = OUTPUT_DIR / 'char_vocab.json'
    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)
    print(f"Char vocab saved → {char_vocab_path}  ({char_vocab['vocab_size']} tokens)")

    vocab_path = OUTPUT_DIR / 'georgian_vocabulary.txt'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for w in sorted(vocab):
            f.write(f"{w}\n")
    print(f"Word list saved  → {vocab_path}  ({len(vocab):,} words)")

    n_error    = (df['input_text'] != df['target_text']).sum()
    n_identity = len(df) - n_error

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lengths = df['target_text'].str.len()

    axes[0].hist(lengths, bins=range(MIN_WORD_LEN, MAX_WORD_LEN + 2),
                 color='steelblue', edgecolor='white')
    axes[0].set_title('Target word length distribution')
    axes[0].set_xlabel('Characters'); axes[0].set_ylabel('Pairs')

    axes[1].bar(['Error pairs', 'Identity pairs'], [n_error, n_identity],
                color=['tomato', 'mediumseagreen'], edgecolor='white')
    axes[1].set_title('Dataset composition'); axes[1].set_ylabel('Pairs')

    top_words  = [w for w, _ in word_freq.most_common(20)]
    top_counts = [word_freq[w] for w in top_words]
    axes[2].barh(top_words[::-1], top_counts[::-1], color='slateblue', edgecolor='white')
    axes[2].set_title('Top 20 most frequent words')
    axes[2].set_xlabel('Frequency in dump')

    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'dataset_stats.png'
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Plot saved       → {plot_path}")


# ── Step 8: Print summary ─────────────────────────────────────────────────────

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
    print(f"  Special token indices: PAD={char_vocab['PAD_IDX']} "
          f"SOS={char_vocab['SOS_IDX']} EOS={char_vocab['EOS_IDX']} "
          f"UNK={char_vocab['UNK_IDX']}")
    print("=" * 55)
    print()
    print("Sample pairs (10 random):")
    print(f"  {'INPUT':<22} {'TARGET':<22} TYPE")
    print("  " + "-" * 52)
    for _, row in df.sample(10, random_state=0).iterrows():
        kind = "identity" if row['input_text'] == row['target_text'] else "error"
        print(f"  {row['input_text']:<22} {row['target_text']:<22} {kind}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    if not WIKI_DUMP_PATH.exists():
        print(f"ERROR: Wiki dump not found at {WIKI_DUMP_PATH}")
        sys.exit(1)

    word_freq  = build_word_frequency(WIKI_DUMP_PATH)
    vocab      = filter_vocabulary(word_freq)

    if len(vocab) < 1000:
        print(f"WARNING: only {len(vocab)} words passed the frequency filter. "
              f"Consider lowering MIN_WORD_FREQ (currently {MIN_WORD_FREQ}).")

    df         = build_dataset(vocab, word_freq)
    verify_dataset(df)      # ← new quality gate: prints edit-dist distribution
    char_vocab = build_char_vocab(df)
    save_outputs(df, vocab, char_vocab, word_freq)
    print_summary(df, char_vocab)