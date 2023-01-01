import spacy


class Tagger(object):

    def __init__(self):
        self._spacy_tagger = spacy.load('en_core_web_sm',disable=['parser','ner'])

    def tokenize_text(self, text: str):

        return [t for t in self._spacy_tagger(text)]
