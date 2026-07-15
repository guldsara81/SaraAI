from saraai.datasets.text_dataset import TextDataset


def test_dataset():

    dataset = TextDataset()

    dataset.add_text("Hej världen")

    dataset.add_text("SaraAI är kul")

    assert dataset.size() == 2

    assert dataset.get_texts()[0] == "Hej världen"