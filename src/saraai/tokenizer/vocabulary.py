"""
SaraAI Vocabulary

Ansvarar för att lagra och hantera ordlistan
som används av tokenizern.
"""


class Vocabulary:
    """Representerar modellens ordförråd."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

        self._add_special_tokens()

    def _add_special_tokens(self):
        """Lägger till specialtokens."""

        for token in (
            self.PAD,
            self.UNK,
            self.BOS,
            self.EOS,
        ):
            self.add_word(token)

    def add_word(self, word: str) -> int:
        """
        Lägger till ett ord i vokabulären.

        Returnerar ordets ID.
        """

        if word not in self.word_to_id:
            index = len(self.word_to_id)

            self.word_to_id[word] = index
            self.id_to_word[index] = word

        return self.word_to_id[word]

    def get_id(self, word: str) -> int:
        """Returnerar ordets ID."""

        return self.word_to_id.get(
            word,
            self.word_to_id[self.UNK],
        )

    def get_word(self, index: int) -> str:
        """Returnerar ordet från ett ID."""

        return self.id_to_word.get(
            index,
            self.UNK,
        )

    def size(self) -> int:
        """Storleken på vokabulären."""

        return len(self.word_to_id)