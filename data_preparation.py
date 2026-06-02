import fitz
import re
import random
import pandas as pd
import bz2
import xml.etree.ElementTree as ET
from collections import Counter

# Constants
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
    'კ': ['ჯ', 'ი', 'ო', 'ლ', 'მ'],
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

PHONETIC_PAIRS = {
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

ERROR_TYPE_WEIGHTS = [
    ('keyboard_swap', 0.40),
    ('transposition', 0.25),
    ('omission',      0.20),
    ('insertion',     0.10),
    ('phonetic_swap', 0.05),
]
ERROR_TYPES, ERROR_PROBS = zip(*ERROR_TYPE_WEIGHTS)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


def extract_words_from_wikipedia_dump(dump_path, max_unique_words=80000):
    """
    FIXED ISSUE 3: Processes and filters words on the fly directly inside the
    XML stream loop instead of creating a massive raw string in memory.
    """
    RE_TEMPLATE    = re.compile(r'\{\{.*?\}\}', re.DOTALL)
    RE_LINK        = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')
    RE_TAG         = re.compile(r'<[^>]+>')
    RE_HEADING     = re.compile(r'={2,}.*?={2,}')
    RE_PUNCTUATION = re.compile(r'[^\u10D0-\u10FA\s]')

    seen_words = set()
    clean_words = []

    with bz2.open(dump_path, 'rb') as f:
        ns = 'http://www.mediawiki.org/xml/export-0.11/'
        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag == f'{{{ns}}}text' and elem.text:
                raw = elem.text
                raw = RE_TEMPLATE.sub(' ', raw)
                raw = RE_LINK.sub(r'\1', raw)
                raw = RE_TAG.sub(' ', raw)
                raw = RE_HEADING.sub(' ', raw)
                raw = RE_PUNCTUATION.sub(' ', raw)

                # Tokenize and clean words on the fly to save RAM
                for token in raw.split():
                    word = token.strip('.,!?"()[]{}«»:-—;""„…')
                    if re.match(r'^[\u10D0-\u10FA]{3,}$', word):
                        seen_words.add(word)
                        clean_words.append(word)

                elem.clear()

                if max_unique_words and len(seen_words) >= max_unique_words:
                    break

    return clean_words


def tokenize_and_clean_georgian(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    clean_words = []
    for token in text.split(' '):
        word = token.strip('.,!?"()[]{}«»:-—;""„…')
        if re.match(r'^[\u10D0-\u10FA]{3,}$', word):
            clean_words.append(word)
    return clean_words


def inject_synthetic_error(word, num_errors=1):
    if len(word) < 3:
        return word

    char_list = list(word)

    for _ in range(num_errors):
        if len(char_list) < 3:
            break

        error_type = random.choices(ERROR_TYPES, weights=ERROR_PROBS, k=1)[0]
        idx = random.randint(0, len(char_list) - 1)

        if error_type == 'omission':
            char_list.pop(idx)

        elif error_type == 'insertion':
            char = char_list[idx]
            if char in GEORGIAN_KEYBOARD_NEIGHBORS:
                neighbor = random.choice(GEORGIAN_KEYBOARD_NEIGHBORS[char])
            else:
                neighbor = char
            char_list.insert(idx + 1, neighbor)

        elif error_type == 'keyboard_swap':
            char = char_list[idx]
            if char in GEORGIAN_KEYBOARD_NEIGHBORS:
                char_list[idx] = random.choice(GEORGIAN_KEYBOARD_NEIGHBORS[char])
            else:
                # FIXED ISSUE 2: Replaced character removal fallback with a random valid key
                # to prevent substitution errors from masquerading as omissions.
                char_list[idx] = random.choice(list(GEORGIAN_KEYBOARD_NEIGHBORS.keys()))

        elif error_type == 'phonetic_swap':
            char = char_list[idx]
            if char in PHONETIC_PAIRS:
                char_list[idx] = PHONETIC_PAIRS[char]

        elif error_type == 'transposition':
            # FIXED ISSUE 1: Adjusted boundary evaluation. If the random index hits the
            # absolute end of the array, it shifts down to perform a clean swap backward.
            if idx == len(char_list) - 1:
                idx -= 1
            char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]

    return "".join(char_list)


def build_parallel_dataset(word_list, variants_per_word=3):
    word_freq = Counter(word_list)
    unique_words = list(word_freq.keys())

    max_freq = max(word_freq.values())
    min_variants = variants_per_word
    max_variants = variants_per_word + 3

    dataset = []

    for word in unique_words:
        freq_ratio = word_freq[word] / max_freq
        word_variants = min_variants + round(freq_ratio * (max_variants - min_variants))

        dataset.append({'input_text': word, 'target_text': word})

        corrupted_seen = set()
        attempts = 0
        generated = 0
        target_corrupted = word_variants - 1

        max_errors = max(1, min(len(word) // 4, 3))

        while generated < target_corrupted and attempts < target_corrupted * 6:
            attempts += 1
            num_errors = random.randint(1, max_errors)
            corrupted = inject_synthetic_error(word, num_errors=num_errors)

            if corrupted != word and corrupted not in corrupted_seen and len(corrupted) >= 2:
                corrupted_seen.add(corrupted)
                dataset.append({'input_text': corrupted, 'target_text': word})
                generated += 1

    df = pd.DataFrame(dataset)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    random.seed(42)

    pdf_filename  = "reference/დათა-თუთაშხია.pdf"
    wiki_filename = r"C:\Users\gigic\Downloads\kawiki-20251020-pages-articles-multistream.xml.bz2"

    raw_text_novel = extract_text_from_pdf(pdf_filename)
    # Patched caller to directly collect processed word tokens
    words_wiki  = extract_words_from_wikipedia_dump(wiki_filename, max_unique_words=80000)
    words_novel = tokenize_and_clean_georgian(raw_text_novel)

    all_words = words_novel + words_wiki

    unique_vocabulary = sorted(set(all_words))
    with open("data/georgian_vocabulary.txt", "w", encoding="utf-8") as f:
        for word in unique_vocabulary:
            f.write(f"{word}\n")

    dataset_df = build_parallel_dataset(all_words, variants_per_word=3)
    dataset_df.to_csv("data/georgian_spellcheck_dataset.csv", index=False, encoding="utf-8")

    error_pairs = dataset_df[dataset_df['input_text'] != dataset_df['target_text']]
    clean_pairs = dataset_df[dataset_df['input_text'] == dataset_df['target_text']]

    print(f"Total rows:         {len(dataset_df)}")
    print(f"Unique target words:{dataset_df['target_text'].nunique()}")
    print(f"Clean pairs:        {len(clean_pairs)} ({100*len(clean_pairs)/len(dataset_df):.1f}%)")
    print(f"Error pairs:        {len(error_pairs)} ({100*len(error_pairs)/len(dataset_df):.1f}%)")