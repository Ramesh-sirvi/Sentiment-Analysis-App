"""
preprocessing.py
=================
Text cleaning pipeline for the Twitter sentiment BERT model.

IMPORTANT: This mirrors, step-for-step, the cleaning pipeline that was
applied to `twitter_training.csv` in Twitter_sentiment_analysis.ipynb
before the model was fine-tuned. Inference-time preprocessing must
match training-time preprocessing or the model will see out-of-
distribution input and predictions will be unreliable.

Notebook pipeline (applied to the training dataframe, in this order):
    1. remove_html_tags   -> strip anything that looks like an HTML tag
    2. remove_urls        -> strip http(s):// links
    3. remove_punctuation -> strip all string.punctuation characters
    4. handling_emoji     -> strip emoji characters
    5. text_process        -> strip English stopwords (+ a few SMS-speak
                              tokens the notebook added: u, ur, 4, 2, im,
                              dont, doin, ure)

Note: the notebook's `remove_newline` step was only ever applied to the
validation dataframe (df2), not to the training dataframe (df) that the
model actually learned from — so it is intentionally left out here to
stay faithful to what the model was trained on. Newlines are still
implicitly collapsed by the stopword-removal step, which rejoins the
text on single spaces.
"""

import re
import string

import nltk
from nltk.corpus import stopwords

_NLTK_RESOURCES = [
    "stopwords",
    "punkt",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
]


def ensure_nltk_resources() -> None:
    """Download the NLTK resources the notebook used, if not already present."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


def remove_html_tags(text: str) -> str:
    pattern = r"<.*?>"
    return re.sub(pattern, "", text)


def remove_urls(text: str) -> str:
    pattern = r"https?://\S+"
    return re.sub(pattern, "", text)


_PUNCTUATION = list(string.punctuation)


def remove_punctuation(text: str) -> str:
    for ch in _PUNCTUATION:
        text = text.replace(ch, "")
    return text


def handling_emoji(text: str) -> str:
    import emoji

    for item in emoji.emoji_list(text):
        text = text.replace(item["emoji"], "")
    return text


_EXTRA_STOPWORDS = ["u", "ü", "ur", "4", "2", "im", "dont", "doin", "ure"]


def text_process(mess: str) -> str:
    stop_words = stopwords.words("english") + _EXTRA_STOPWORDS

    # Notebook rebuilds the string without punctuation again here (redundant
    # after remove_punctuation, but kept for parity) before dropping stopwords.
    nopunc = "".join(ch for ch in mess if ch not in string.punctuation)

    return " ".join(word for word in nopunc.split() if word.lower() not in stop_words)


def preprocess_text(text: str) -> str:
    """Run the full training-time cleaning pipeline on a single piece of text."""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = handling_emoji(text)
    text = text_process(text)
    return text
