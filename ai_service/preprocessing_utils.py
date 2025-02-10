import re

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(" +", " ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"^\d+ Contributors", "", text)
    text = re.sub(r"^(.*?)Lyrics", "", text, flags=re.MULTILINE)
    return text


def tokenize(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
    ]


def preprocess(text: str) -> list[str]:
    text = clean(text)
    return tokenize(text)
