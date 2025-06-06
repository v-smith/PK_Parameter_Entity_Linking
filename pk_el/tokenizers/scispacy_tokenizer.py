import spacy

nlp = spacy.load("en_core_sci_sm")

def scispacy_tokenizer(text):
    """Tokenizes text using SciSpacy."""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def scispacy_batch_tokenizer(texts):
    """
    Tokenizes a batch of texts using SciSpacy.

    :param texts: List of strings to tokenize.
    :return: List of tokenized lists (one list per input text).
    """
    tokenized_texts = []

    for doc in nlp.pipe(texts, batch_size=32, disable=["ner", "parser"]):
        tokens = [token.text for token in doc]
        tokenized_texts.append(tokens)

    return tokenized_texts