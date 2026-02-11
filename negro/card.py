from dataclasses import dataclass


@dataclass
class Card:
    num: int
    suit: str

    def __repr__(self) -> str:
        return f"[{self.num} {self.suit}]"


class Joker:
    def __repr__(self) -> str:
        return "[* 🃏]"
