import fitz  # PyMuPDF
import re
import random
import pandas as pd

# 1. GEORGIAN TYPO EMULATION CONFIGURATIONS

# Proximity map based on the standard Georgian QWERTY keyboard layout
# this simulates physical fingers accidentally slipping to adjacent keys (needed for ai training data)
GEORGIAN_KEYBOARD_NEIGHBORS = {
    'ქ': ['წ', 'ა', 'ს', 'უ'],
    'წ': ['ქ', 'ე', 'ს', 'დ', 'ა'],
    'ე': ['წ', 'რ', 'დ', 'ფ', 'ს'],
    'რ': ['ე', 'ტ', 'ფ', 'გ'],
    'ტ': ['რ', 'ყ', 'გ', 'ჰ'],
    'ყ': ['ტ', 'უ', 'ჰ', 'ჯ'],
    'უ': ['ყ', 'ი', 'ჯ', 'ჰ'],
    'ი': ['უ', 'ო', 'ჰ', 'ლ'],
    'ო': ['ი', 'პ', 'ლ', 'შ'],
    'პ': ['ო', 'ლ', 'შ'],
    'ა': ['ქ', 'ს', 'ზ'],
    'ს': ['ა', 'დ', 'ზ', 'ხ'],
    'დ': ['ს', 'ფ', 'ხ', 'ც'],
    'ფ': ['დ', 'გ', 'ც', 'ვ'],
    'გ': ['ფ', 'ჰ', 'ვ', 'ბ'],
    'ჰ': ['გ', 'ჯ', 'ბ', 'ნ'],
    'ჯ': ['ჰ', 'ლ', 'ნ', 'მ'],
    'ლ': ['ჯ', 'შ', 'მ'],
    'შ': ['ლ'],
    'ზ': ['ა', 'ს', 'ხ'],
    'ხ': ['ზ', 'დ', 'ც'],
    'ც': ['ხ', 'ფ', 'ვ'],
    'ვ': ['ც', 'გ', 'ბ'],
    'ბ': ['ვ', 'ჰ', 'ნ'],
    'ნ': ['ბ', 'ჯ', 'მ'],
    'მ': ['ნ', 'ჯ', 'ლ']
}

# Phonetic similarity configuration
# Simulates human spelling confusion or OCR reading mistakes
PHONETIC_PAIRS = {
    'კ': 'ქ', 'ქ': 'კ',
    'ტ': 'თ', 'თ': 'ტ',
    'პ': 'ფ', 'ფ': 'პ',
    'ც': 'წ', 'წ': 'ც',
    'ჩ': 'ჭ', 'ჭ': 'ჩ',
    'ძ': 'ც', 'თ': 'ტ'
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


def tokenize_and_clean_georgian(text):
    #Splits raw text into a flat list of valid, clean Georgian words
    print("[*] Cleaning and tokenizing text into Georgian words...")
    # Normalize whitespaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    raw_tokens = text.split(' ')

    clean_words = []
    for token in raw_tokens:
        # Strip trailing and leading punctuation marks commonly used in texts
        word = token.strip('.,!?"()[]{}«»:-—;“”„…')

        # Keep word strictly if it consists purely of Georgian Mkhedruli letters (\u10D0-\u10FA)
        if re.match(r'^[\u10D0-\u10FA]+$', word):
            clean_words.append(word)

    return clean_words


def inject_synthetic_error(word):
    #Introduces a single random typo into a given Georgian word
    # dont try to mess up words that are too short, or they lose all phonetic context
    if len(word) <= 2:
        return word

    error_type = random.choice(['omission', 'insertion', 'keyboard_swap', 'phonetic_swap', 'transposition'])
    idx = random.randint(0, len(word) - 1)
    char_list = list(word)

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

    elif error_type == 'phonetic_swap':
        # Confuse an aspirate/glottal stop pair or closely-related sound
        char = char_list[idx]
        if char in PHONETIC_PAIRS:
            char_list[idx] = PHONETIC_PAIRS[char]

    elif error_type == 'transposition':
        # Fast typing error where adjacent letters swap orders (e.g. წერილი -> წერილაი)
        if idx < len(word) - 1:
            char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]

    return "".join(char_list)


def build_parallel_dataset(word_list, corruption_rate=0.5):
    """Creates an AI dataset with balanced correct and corrupted text inputs."""
    print(f"[*] Building dataset matrix (Corruption Rate: {corruption_rate * 100}%)...")
    dataset = []

    for word in word_list:
        # Decide whether to corrupt this row or leave it correct
        if random.random() < corruption_rate:
            input_word = inject_synthetic_error(word)
        else:
            input_word = word

        dataset.append({
            'input_word': input_word,
            'target_word': word
        })

    # Convert to standard Pandas DataFrame
    df = pd.DataFrame(dataset)
    # Deduplicate matching pairs to keep dataset diverse and clean
    df = df.drop_duplicates().reset_index(drop=True)
    return df

# 3. RUNTIME EXECUTION PIPELINE

if __name__ == "__main__":
    pdf_filename = "reference/დათა-თუთაშხია.pdf"

    # Step 1: Read the book PDF
    raw_text = extract_text_from_pdf(pdf_filename)
    print(f"[✓] Successfully extracted {len(raw_text)} raw characters.")

    # Step 2: Split text down into valid vocabulary items
    all_words = tokenize_and_clean_georgian(raw_text)
    print(f"[✓] Extracted a total corpus of {len(all_words)} parsed words.")

    # Step 3: Save a clean, distinct dictionary file (Only unique correct words)
    unique_vocabulary = sorted(list(set(all_words)))
    vocab_filename = "georgian_vocabulary.txt"
    with open(vocab_filename, "w", encoding="utf-8") as f:
        for word in unique_vocabulary:
            f.write(f"{word}\n")
    print(f"[✓] Saved dictionary of {len(unique_vocabulary)} unique words to '{vocab_filename}'")

    # Build training mapping dataset (50% correct, 50% typos)
    # We pass the full sequential word flow to capture structural duplicates and natural word frequency
    dataset_df = build_parallel_dataset(all_words, corruption_rate=0.5)

    # Save as CSV ready for AI framework input mapping
    csv_filename = "georgian_spellchecker_dataset.csv"
    dataset_df.to_csv(csv_filename, index=False, encoding="utf-8")

    print(f"[✓] Training dataset compiled into '{csv_filename}' with {len(dataset_df)} total parallel training rows!")