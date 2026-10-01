"""
Quick sanity checks for preprocessing.py — run with:
    python -m pytest test_preprocessing.py -v
or just:
    python test_preprocessing.py
"""

from preprocessing import (
    ensure_nltk_resources,
    handling_emoji,
    remove_html_tags,
    remove_punctuation,
    remove_urls,
    text_process,
)

ensure_nltk_resources()


def test_remove_html_tags():
    assert remove_html_tags("<b>great</b> movie") == "great movie"


def test_remove_urls():
    assert remove_urls("check this out https://example.com now") == "check this out  now"


def test_remove_punctuation():
    assert remove_punctuation("wow!! amazing...") == "wow amazing"


def test_handling_emoji():
    cleaned = handling_emoji("I love this game 😍🔥")
    assert "😍" not in cleaned and "🔥" not in cleaned


def test_text_process_drops_stopwords():
    out = text_process("this is the best game i have ever played")
    # "this", "is", "the", "i", "have" are stopwords and should be dropped
    for stopword in ["this", "is", "the"]:
        assert stopword not in out.split()
    assert "best" in out and "game" in out and "played" in out


if __name__ == "__main__":
    test_remove_html_tags()
    test_remove_urls()
    test_remove_punctuation()
    test_handling_emoji()
    test_text_process_drops_stopwords()
    print("All preprocessing sanity checks passed.")
