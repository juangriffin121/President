from card import Card, Joker
from strategy import Strategy
from itertools import combinations
from collections import defaultdict


class Player:
    def __init__(self, name: str, strategy: Strategy) -> None:
        self.name = name
        self.hand = []
        self.strategy = strategy
        self.id = 0

    def choose_cards(self, global_state) -> list[Card | Joker] | None:
        choice = self.strategy.choose_cards(global_state, self.hand)
        self.drop_cards(choice)
        return choice

    def drop_cards(self, cards: list[Card | Joker] | None):
        self.hand = list(filter(lambda card: card not in cards, self.hand))

    def __repr__(self) -> str:
        return f"{self.name}:\n\t{self.hand.__repr__()}\n"

    def order_hand(self) -> dict[int, list[list[Card | Joker]]]:
        result = defaultdict(list)

        cards = [c for c in self.hand if not isinstance(c, Joker)]
        jokers = [c for c in self.hand if isinstance(c, Joker)]

        nums = set(c.num for c in cards)

        for n in range(1, len(self.hand) + 1):
            for num in nums:
                matching = [c for c in cards if c.num == num]

                for k in range(min(len(jokers), n) + 1):
                    needed = n - k
                    if needed <= len(matching):
                        for combo in combinations(matching, needed):
                            result[n].append(list(combo) + jokers[:k])

        return dict(result)
