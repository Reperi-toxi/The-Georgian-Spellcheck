import fitz  # PyMuPDF
import re
import random
import pandas as pd
import bz2
import xml.etree.ElementTree as ET

# 1. GEORGIAN TYPO EMULATION CONFIGURATIONS

# Proximity map based on the standard Georgian QWERTY keyboard layout
# this simulates physical fingers accidentally slipping to adjacent keys (needed for ai training data)
# caveat: we now also include such examples as 'neighbours': წ -> (shift + წ =) ჭ...
GEORGIAN_KEYBOARD_NEIGHBORS = {
    'ქ': ['წ', 'ა', 'ს'],
    'წ': ['ქ', 'ე', 'ს', 'დ', 'ა', 'ჭ'],
    'ჭ': ['ქ', 'ე', 'ს', 'დ', 'ა', 'წ'],
    'ე': ['წ', 'რ', 'დ', 'ფ', 'ს'],
    'რ': ['ე', 'ტ', 'ფ', 'გ', 'დ', 'ღ'],
    'ღ': ['ე', 'ტ', 'ფ', 'გ', 'დ', 'რ'],
    'ტ': ['რ', 'ყ', 'გ', 'ჰ', 'ფ', 'თ'],
    'თ': ['რ', 'ყ', 'გ', 'ჰ', 'ფ', 'ტ'],
    'ყ': ['ტ', 'უ', 'ჰ', 'ჯ', 'გ'],
    'უ': ['ყ', 'ი', 'ჯ', 'ჰ', 'კ'],
    'ი': ['უ', 'ო', 'ჯ', 'ლ', 'კ'],
    'ო': ['ი', 'პ', 'ლ', 'კ'],
    'პ': ['ო', 'ლ'],
    'ა': ['ქ', 'ს', 'ზ', 'წ'],
    'ს': ['ა', 'დ', 'ზ', 'ხ', 'წ', 'ქ', 'ე', 'შ'],
    'შ': ['ა', 'დ', 'ზ', 'ხ', 'წ', 'ქ', 'ე', 'ს'],
    'დ': ['ს', 'ფ', 'ხ', 'ც', 'ე'],
    'ფ': ['დ', 'გ', 'ც', 'ვ', 'რ', 'ტ'],
    'გ': ['ფ', 'ჰ', 'ვ', 'ბ', 'ტ', 'ყ'],
    'ჰ': ['გ', 'ჯ', 'ბ', 'ნ', 'ყ', 'უ'],
    'ჯ': ['ჰ', 'კ', 'ნ', 'მ', 'უ', 'ი', 'ჟ'],
    'ჟ': ['ჰ', 'კ', 'ნ', 'მ', 'უ', 'ი', 'ჯ'],
    'კ': ['ჯ','ი','ო','ლ','მ'],
    'ლ': ['კ', 'ო', 'პ'],
    'ზ': ['ა', 'ს', 'ხ', 'ძ'],
    'ძ': ['ა', 'ს', 'ხ', 'ზ'],
    'ხ': ['ზ', 'დ', 'ც', 'ს'],
    'ც': ['ხ', 'ფ', 'ვ', 'დ', 'ჩ'],
    'ჩ': ['ხ', 'ფ', 'ვ', 'დ', 'ც'],
    'ვ': ['ც', 'გ', 'ბ', 'ფ'],
    'ბ': ['ვ', 'ჰ', 'ნ', 'გ'],
    'ნ': ['ბ', 'ჯ', 'მ', 'ჰ'],
    'მ': ['ნ', 'ჯ', 'კ'],
}

# Phonetic similarity configuration
# Simulates human spelling confusion or OCR reading mistakes
PHONETIC_PAIRS = {
    # Ejective vs aspirate stops — the most common Georgian spelling confusion
    'კ': 'ქ', 'ქ': 'კ',
    'ტ': 'თ', 'თ': 'ტ',
    'პ': 'ფ', 'ფ': 'პ',
    'ც': 'წ', 'წ': 'ც',
    'ჩ': 'ჭ', 'ჭ': 'ჩ',
    'ძ': 'ზ', 'ზ': 'ძ',   # voiced affricate vs fricative
    'ღ': 'გ', 'გ': 'ღ',   # uvular vs velar
    'ხ': 'ჰ', 'ჰ': 'ხ',   # velar vs glottal fricative
    'შ': 'ს', 'ს': 'შ',   # palatal vs alveolar sibilant
    'ჟ': 'ზ', 'ზ': 'ჟ',   # voiced palatal vs alveolar
    'რ': 'ლ', 'ლ': 'რ',   # liquids
}

# 2. CORE UTILITY FUNCTIONS

def extract_text_from_pdf(pdf_path):
    """Extracts raw text strings from all pages of the target PDF."""
    print(f"[*] Extracting text from {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[!] Error opening file. Make sure '{pdf_path}' is in the current directory.")
        raise e

    full_text = []
    for page in doc:
        text = page.get_text("text")
        full_text.append(text)
    return "\n".join(full_text)


def extract_text_from_wikipedia_dump(dump_path, max_unique_words=80000):
    """
    Extracts raw Georgian text from a Wikipedia XML dump (.xml.bz2).
    Streams the file to avoid loading 223MB into memory all at once.
    Strips all wiki markup, templates, and XML tags — returns plain text.

    Args:
        dump_path:        Path to the .xml.bz2 file downloaded from dumps.wikimedia.org
        max_unique_words: Stops once this many unique Georgian words have been collected.
                          Controls dataset size without guessing article counts.
                          Set to None to process the full dump.
    """
    print(f"[*] Extracting text from Wikipedia dump: {dump_path}...")

    # Regex patterns to strip Wikipedia markup before tokenizing
    RE_TEMPLATE    = re.compile(r'\{\{.*?\}\}', re.DOTALL)          # {{templates}}
    RE_LINK        = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')  # [[links]] → keep label
    RE_TAG         = re.compile(r'<[^>]+>')                          # <xml tags>
    RE_HEADING     = re.compile(r'={2,}.*?={2,}')                    # == Section headings ==
    RE_PUNCTUATION = re.compile(r'[^\u10D0-\u10FA\s]')              # keep only Georgian + whitespace

    full_text_parts = []
    article_count = 0
    seen_words = set()

    # Use iterparse to stream-parse the bz2 file without full decompression into RAM
    with bz2.open(dump_path, 'rb') as f:
        ns = 'http://www.mediawiki.org/xml/export-0.11/'

        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag == f'{{{ns}}}text' and elem.text:
                raw = elem.text

                # Strip wiki markup layer by layer
                raw = RE_TEMPLATE.sub(' ', raw)
                raw = RE_LINK.sub(r'\1', raw)
                raw = RE_TAG.sub(' ', raw)
                raw = RE_HEADING.sub(' ', raw)
                raw = RE_PUNCTUATION.sub(' ', raw)

                # Track unique words to enforce the cap
                for token in raw.split():
                    word = token.strip('.,!?"()[]{}«»:-—;""„…')
                    if re.match(r'^[\u10D0-\u10FA]+$', word):
                        seen_words.add(word)

                full_text_parts.append(raw)
                article_count += 1

                # Free the element from memory immediately (critical for large dumps)
                elem.clear()

                # Stop once we have enough unique words
                if max_unique_words and len(seen_words) >= max_unique_words:
                    print(f"[✓] Reached {len(seen_words)} unique words after {article_count} articles. Stopping.")
                    break

    print(f"[✓] Processed {article_count} Wikipedia articles.")
    return "\n".join(full_text_parts)


def tokenize_and_clean_georgian(text):
    """Splits raw text into a flat list of valid, clean Georgian words."""
    print("[*] Cleaning and tokenizing text into Georgian words...")
    text = text.replace('\n', ' ').replace('\t', ' ')
    raw_tokens = text.split(' ')

    clean_words = []
    for token in raw_tokens:
        word = token.strip('.,!?"()[]{}«»:-—;""„…')
        if re.match(r'^[\u10D0-\u10FA]+$', word):
            clean_words.append(word)

    return clean_words


def inject_synthetic_error(word, num_errors=1):
    """
    Introduces one or more random typos into a given Georgian word.

    FIX: previously always injected exactly 1 error, and skipped words <=2 chars entirely.
    Now:
      - Short words (len==1) are still skipped — a 1-char word can't be meaningfully corrupted
      - Words of length 2-3 get at most 1 error
      - Words of length 4+ can receive 1 or 2 errors based on num_errors argument
      - keyboard_swap and phonetic_swap now fall back to omission if the character
        has no defined neighbor/pair, so the error type is never silently wasted
    """
    if len(word) <= 1:
        return word

    # Cap errors based on word length to avoid making words unrecognizable
    max_errors = 1 if len(word) <= 3 else num_errors
    actual_errors = random.randint(1, max_errors)

    char_list = list(word)

    for _ in range(actual_errors):
        if len(char_list) <= 1:
            break

        error_type = random.choice(['omission', 'insertion', 'keyboard_swap', 'phonetic_swap', 'transposition'])
        idx = random.randint(0, len(char_list) - 1)

        if error_type == 'omission':
            # Drop a letter completely (e.g. ქართული -> ქართლი)
            char_list.pop(idx)

        elif error_type == 'insertion':
            # Accidentally double-strike a key (e.g. წყალი -> წყაალი)
            char_list.insert(idx, char_list[idx])

        elif error_type == 'keyboard_swap':
            # Hit a physically neighboring letter on the keyboard
            char = char_list[idx]
            if char in GEORGIAN_KEYBOARD_NEIGHBORS:
                char_list[idx] = random.choice(GEORGIAN_KEYBOARD_NEIGHBORS[char])
            else:
                char_list.pop(idx)

        elif error_type == 'phonetic_swap':
            # Confuse phonetically similar characters
            char = char_list[idx]
            if char in PHONETIC_PAIRS:
                char_list[idx] = PHONETIC_PAIRS[char]
            else:
                if idx < len(char_list) - 1:
                    char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]

        elif error_type == 'transposition':
            # Fast typing error where adjacent letters swap orders
            if idx < len(char_list) - 1:
                char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]

    return "".join(char_list)

def build_parallel_dataset(word_list, corruption_rate=0.5, variants_per_word=3):
    """
    Creates an AI dataset with balanced correct and corrupted text inputs.

    FIX: previously each word produced exactly 1 training pair (either clean or corrupted).
    Now each unique word produces multiple pairs:
      - 1 guaranteed clean pair (input == target) so the model always learns to pass through correct words
      - (variants_per_word - 1) corrupted variants with different random errors
    This gives the model exposure to multiple error types per word, not just one.

    Args:
        word_list:         Full sequential word list (duplicates preserved for frequency weighting)
        corruption_rate:   Kept for compatibility but variants approach supersedes it
        variants_per_word: How many training pairs to generate per unique word (default 3:
                           1 clean + 2 corrupted variants)
    """
    print(f"[*] Building dataset matrix ({variants_per_word} variants per word)...")
    dataset = []

    unique_words = list(set(word_list))

    for word in unique_words:
        # Always include one clean pair — teaches model to leave correct words alone
        dataset.append({'input_word': word, 'target_word': word})

        # Generate (variants_per_word - 1) distinct corrupted versions
        corrupted_seen = set()
        attempts = 0
        generated = 0
        target_corrupted = variants_per_word - 1

        while generated < target_corrupted and attempts < target_corrupted * 4:
            attempts += 1
            # Allow double errors for longer words to increase variety
            max_err = 2 if len(word) >= 5 else 1
            num_errors = random.randint(1, max_err)
            corrupted = inject_synthetic_error(word, num_errors=num_errors)

            # Skip if corruption produced no change or a duplicate variant
            if corrupted != word and corrupted not in corrupted_seen:
                corrupted_seen.add(corrupted)
                dataset.append({'input_word': corrupted, 'target_word': word})
                generated += 1

    df = pd.DataFrame(dataset)
    df = df.drop_duplicates().reset_index(drop=True)
    # Shuffle so novel and wikipedia words are interleaved
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# 3. RUNTIME EXECUTION PIPELINE

if __name__ == "__main__":
    pdf_filename  = "reference/დათა-თუთაშხია.pdf"
    wiki_filename = r"C:\Users\gigic\Downloads\kawiki-20251020-pages-articles-multistream.xml.bz2"

    # --- Step 1: Extract text from both sources ---

    raw_text_novel = extract_text_from_pdf(pdf_filename)
    print(f"[✓] Novel: extracted {len(raw_text_novel)} raw characters.")

    # Pulls articles until 80K unique Georgian words are collected, then stops.
    raw_text_wiki = extract_text_from_wikipedia_dump(wiki_filename, max_unique_words=80000)
    print(f"[✓] Wikipedia: extracted {len(raw_text_wiki)} raw characters.")

    # --- Step 2: Tokenize both sources independently then merge ---

    words_novel = tokenize_and_clean_georgian(raw_text_novel)
    print(f"[✓] Novel corpus: {len(words_novel)} words ({len(set(words_novel))} unique).")

    words_wiki = tokenize_and_clean_georgian(raw_text_wiki)
    print(f"[✓] Wikipedia corpus: {len(words_wiki)} words ({len(set(words_wiki))} unique).")

    all_words = words_novel + words_wiki
    print(f"[✓] Combined corpus: {len(all_words)} total words.")

    # --- Step 3: Save merged vocabulary (unique correct words from both sources) ---

    unique_vocabulary = sorted(list(set(all_words)))
    vocab_filename = "data/georgian_vocabulary.txt"
    with open(vocab_filename, "w", encoding="utf-8") as f:
        for word in unique_vocabulary:
            f.write(f"{word}\n")
    print(f"[✓] Saved merged vocabulary of {len(unique_vocabulary)} unique words to '{vocab_filename}'")

    # --- Step 4: Build training dataset from combined corpus ---

    # Each unique word gets 3 pairs: 1 clean + 2 corrupted variants
    # This produces ~3x the rows of the old approach with much better error coverage
    dataset_df = build_parallel_dataset(all_words, corruption_rate=0.5, variants_per_word=3)

    csv_filename = "data/georgian_spellchecker_dataset.csv"
    dataset_df.to_csv(csv_filename, index=False, encoding="utf-8")

    print(f"[✓] Training dataset compiled into '{csv_filename}' with {len(dataset_df)} total parallel training rows!")
    print(f"    └─ Novel contribution:      {len(set(words_novel))} unique words")
    print(f"    └─ Wikipedia contribution:  {len(set(words_wiki))} unique words")
    print(f"    └─ Total unique words:      {len(unique_vocabulary)}")
    print(f"    └─ Approx pairs per word:   3 (1 clean + 2 corrupted)")