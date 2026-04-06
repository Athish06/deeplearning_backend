"""
Text preprocessing and tokenization pipeline for sentiment analysis.
Handles cleaning, tokenization, and sequence preparation.
Pure Python/NumPy — no framework dependency.
"""

import re
import pickle
import json
from collections import Counter

# ─── Constants ───────────────────────────────────────────────
VOCAB_SIZE = 35000  # Increased from 20000 to capture Yelp slang
MAX_LEN = 200
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
PAD_IDX = 0
OOV_IDX = 1

# Words that negate the meaning of the following word
NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "without", "barely", "hardly",
    "scarcely", "seldom", "rarely",
    # Contractions (after lowercasing and punctuation separation)
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
    "doesn", "don", "didn", "won", "wouldn", "shouldn",
    "couldn", "mustn", "needn",
    "aint", "cant", "wont", "didnt", "doesnt", "dont",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "wouldnt", "shouldnt", "couldnt",
])


def handle_negations(text: str) -> str:
    """
    Fuse negation words with the following word into compound tokens.
    'not good' → 'not_good', 'never seen' → 'never_seen'
    This lets the model learn these as single sentiment-bearing units.
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if words[i] in NEGATION_WORDS and i + 1 < len(words):
            # Only fuse with alphabetic words (skip punctuation)
            next_word = words[i + 1]
            if next_word.isalpha():
                result.append(f"{words[i]}_{next_word}")
                i += 2
                continue
        result.append(words[i])
        i += 1
    return " ".join(result)


def clean_text(text: str) -> str:
    """
    Clean raw text for model input.
    - Lowercase
    - Strip HTML tags
    - Remove URLs and emails
    - Normalize whitespace
    - Retain meaningful punctuation (!, ?, .)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # Add spaces around punctuation so they become distinct tokens
    text = re.sub(r"([!?.,'])", r" \1 ", text)

    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"([!?.])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Fuse negation bigrams: "not good" → "not_good"
    text = handle_negations(text)

    return text


class SimpleTokenizer:
    """
    Simple word-level tokenizer (replaces Keras Tokenizer).
    Builds vocabulary from training data and converts text to integer sequences.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word_to_idx = {PAD_TOKEN: PAD_IDX, OOV_TOKEN: OOV_IDX}
        self.idx_to_word = {PAD_IDX: PAD_TOKEN, OOV_IDX: OOV_TOKEN}
        self._fitted = False

    def fit(self, texts: list) -> "SimpleTokenizer":
        """Build vocabulary from list of texts."""
        counter = Counter()
        for text in texts:
            words = text.split()
            counter.update(words)

        # Take top vocab_size - 2 words (reserve PAD and OOV)
        most_common = counter.most_common(self.vocab_size - 2)

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self._fitted = True
        return self

    def text_to_sequence(self, text: str) -> list:
        """Convert a single text to integer sequence."""
        words = text.split()
        return [self.word_to_idx.get(w, OOV_IDX) for w in words]

    def texts_to_sequences(self, texts: list) -> list:
        """Convert list of texts to list of integer sequences."""
        return [self.text_to_sequence(t) for t in texts]

    @property
    def actual_vocab_size(self) -> int:
        return len(self.word_to_idx)


def pad_sequences(sequences: list, maxlen: int = MAX_LEN) -> list:
    """Pad/truncate sequences to fixed length."""
    result = []
    for seq in sequences:
        if len(seq) > maxlen:
            result.append(seq[:maxlen])
        else:
            result.append(seq + [PAD_IDX] * (maxlen - len(seq)))
    return result


def texts_to_padded_sequences(
    texts: list,
    tokenizer: SimpleTokenizer,
    maxlen: int = MAX_LEN,
) -> list:
    """Clean, tokenize, and pad texts in one step."""
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=maxlen)
    return padded


def save_tokenizer(tokenizer: SimpleTokenizer, path: str) -> None:
    """Save tokenizer to disk."""
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: str) -> SimpleTokenizer:
    """Load tokenizer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
