import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # faster

def preprocess_text(text):
    """
    Full NLP preprocessing:
    - tokenization
    - stopword removal
    - punctuation removal
    - lemmatization
    """
    if not isinstance(text, str):
        return ""

    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.like_num:
            continue
        lemma = token.lemma_.strip()
        if lemma:
            tokens.append(lemma)

    return " ".join(tokens)