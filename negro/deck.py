import numpy as np
from negro.card import Card, Joker
import random

# random.seed(1)


class Deck:
    def __init__(self) -> None:
        nums = np.arange(1, 13)
        suits = ["🪙", "🗡️ ", "🏆", "🪵"]
        cards: list[Card | Joker] = [Card(num, suit) for num in nums for suit in suits]
        cards.append(Joker())
        cards.append(Joker())
        self.cards = cards

    def __repr__(self) -> str:
        return [card for card in self.cards.__reversed__()].__repr__()

    def shuffle(self):
        random.shuffle(self.cards)
