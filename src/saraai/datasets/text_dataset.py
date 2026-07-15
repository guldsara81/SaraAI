"""
SaraAI Dataset

Läser text och förbereder träningsdata.
"""

from pathlib import Path


class TextDataset:
    """Läser textfiler för träning."""

    def __init__(self):
        self.texts = []

    def load_file(self, filename: str):
        """
        Läser en textfil.
        """

        path = Path(filename)

        text = path.read_text(
            encoding="utf-8"
        )

        self.texts.append(text)

    def add_text(self, text: str):
        """
        Lägger till text direkt.
        """

        self.texts.append(text)

    def get_texts(self):
        """
        Returnerar all text.
        """

        return self.texts

    def size(self):
        return len(self.texts)