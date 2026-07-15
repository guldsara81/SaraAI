"""
SaraAI Core

Detta är huvudklassen för SaraAI.
All funktionalitet kommer successivt att byggas ut härifrån.
"""


class SaraAI:
    """
    SaraAI huvudklass.
    """

    def __init__(self, name: str = "SaraAI"):
        self.name = name
        self.version = "0.1.0"

    def info(self) -> str:
        """
        Returnerar information om AI:n.
        """
        return (
            f"{self.name} "
            f"(version {self.version})\n"
            "Status: Under utveckling"
        )

    def chat(self, message: str) -> str:
        """
        Enkel platshållare för framtida språkmodell.
        """
        return (
            f"Du skrev: {message}\n\n"
            "Jag är ännu inte tränad, men snart kommer jag kunna prata."
        )