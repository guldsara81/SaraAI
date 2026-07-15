from saraai import SaraAI


def test_name():
    ai = SaraAI()

    assert ai.name == "SaraAI"


def test_info():
    ai = SaraAI()

    assert "SaraAI" in ai.info()


def test_chat():
    ai = SaraAI()

    response = ai.chat("Hej")

    assert "Hej" in response