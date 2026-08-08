def word_tokenize(value):
    return value.split()


class RegexpTokenizer:
    def __init__(self, pattern):
        self.pattern = pattern

    def tokenize(self, value):
        return value.split()


class tokenize:
    RegexpTokenizer = RegexpTokenizer


class data:
    @staticmethod
    def load(name):
        raise AssertionError("the fixture must not use NLTK tokenizers")
