"""
SaraAI Tokenizer
================

Version 0.2

Tokenizern ansvarar endast för att dela upp text och
översätta mellan text och token-ID.

Själva ordlistan hanteras av Vocabulary.
"""

from saraai.tokenizer.vocabulary import Vocabulary


class Tokenizer:
    """Tokenizer som använder en Vocabulary."""

    def __init__(self, vocabulary: Vocabulary | None = None):
        if vocabulary is None:
            vocabulary = Vocabulary()

        self.vocabulary = vocabulary

    def build_vocabulary(self, texts):
        """
        Bygger upp vokabulären från en lista med texter.
        """

        for text in texts:
            for word in text.split():
                self.vocabulary.add_word(word)

    def encode(self, text):
        """
        Gör om text till token-ID.
        """

        return [
            self.vocabulary.get_id(word)
            for word in text.split()
        ]

    def decode(self, token_ids):
        """
        Gör om token-ID tillbaka till text.
        """

        words = [
            self.vocabulary.get_word(token)
            for token in token_ids
        ]

        return " ".join(words)

    @property
    def vocabulary_size(self):
        """
        Returnerar storleken på vokabulären.
        """

        return self.vocabulary.size()