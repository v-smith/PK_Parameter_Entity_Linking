import unicodedata

import nltk

from pk_el.tokenizers.patterns import FRACTION_SLASH_RE, STOP_WORDS_RE, HTML_TAG_RE, PLURAL_RE, BIO_PLURAL_RE, \
    HL_PLURAL_RE

def basic_preprocessing(text: str) -> str:
    """
    Preprocesses the input text by applying normalization, stopword removal, and lemmatization.
    """
    text = text.lower()  # Convert to lowercase
    text = unicodedata.normalize('NFKC', text).replace('\xa0', ' ')  # Normalize Unicode & replace spaces
    text = FRACTION_SLASH_RE.sub("/", text)  # Normalize fraction slashes
    text = STOP_WORDS_RE.sub("", text)  # Remove stop words
    text = HTML_TAG_RE.sub("", text)  # Remove HTML tags
    text = PLURAL_RE.sub("", text)  # Remove plurals
    # Fix specific term plurals
    text = BIO_PLURAL_RE.sub("bioavailability", text)
    text = HL_PLURAL_RE.sub("half life", text)
    return text


def basic_tokenizer(text):
    text = basic_preprocessing(text)
    return text.split()

def nltk_tokenizer(text):
    text = nltk.word_tokenize(text)
    return text




