from saraai.datasets.text_dataset import TextDataset
from saraai.tokenizer.tokenizer import Tokenizer
from saraai.tokenizer.vocabulary import Vocabulary


def _build_dataset(text, sequence_length):
    tokenizer = Tokenizer(Vocabulary())
    tokenizer.build_vocabulary([text])

    dataset = TextDataset(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    dataset.load_text(text)

    return dataset, tokenizer


def test_dataset_length():
    text = "hej världen sara är kul idag"  # 6 tokens

    dataset, _ = _build_dataset(text, sequence_length=2)

    # len = number_of_tokens - sequence_length = 6 - 2 = 4
    assert len(dataset) == 4


def test_dataset_item_is_next_token_shifted():
    text = "hej världen sara är kul idag"

    dataset, tokenizer = _build_dataset(text, sequence_length=2)

    token_ids = tokenizer.encode(text)

    input_tokens, target_tokens = dataset[0]

    assert input_tokens.tolist() == token_ids[0:2]
    assert target_tokens.tolist() == token_ids[1:3]
