from saraai.tokenizer.tokenizer import Tokenizer
from saraai.tokenizer.vocabulary import Vocabulary


def test_tokenizer():

    vocabulary = Vocabulary()

    tokenizer = Tokenizer(vocabulary)

    texts = [
        "hej världen",
        "hej sara",
        "världen är stor"
    ]

    tokenizer.build_vocabulary(texts)

    encoded = tokenizer.encode("hej världen")

    decoded = tokenizer.decode(encoded)

    assert decoded == "hej världen"


def test_unknown_word():

    vocabulary = Vocabulary()

    tokenizer = Tokenizer(vocabulary)

    tokenizer.build_vocabulary(["hej världen"])

    encoded = tokenizer.encode("okänt")

    assert encoded == [vocabulary.get_id("<UNK>")]