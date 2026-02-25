import time
from card import Card, Joker
from state import GlobalState, PlayerState
from strategy import Strategy
from itertools import combinations
from collections import defaultdict

SLEEP_ENABLED = True


def set_sleep_enabled(value: bool) -> None:
    global SLEEP_ENABLED
    SLEEP_ENABLED = value


class Player:
    def __init__(self, name: str, strategy: Strategy) -> None:
        self.name = name
        self.hand = []
        self.strategy = strategy
        self.id = 0

    def choose_cards(self, global_state: GlobalState) -> list[Card | Joker] | None:
        if SLEEP_ENABLED:
            time.sleep(1)
        player_state = PlayerState(self.id, self.hand)
        choice = self.strategy.choose_cards(global_state, player_state)
        # maybe assert choice is valid based on state
        if choice is not None and len(choice) > 0:
            self.drop_cards(choice)
        return choice

    def drop_cards(self, cards: list[Card | Joker] | None):
        self.hand = list(filter(lambda card: card not in cards, self.hand))

    def choose_worst(self, count):
        return self.strategy.choose_worst(count, self.hand)

    def inform_of_results(self, performance: int):
        self.strategy.inform_of_results(performance, self.name)

    def __repr__(self) -> str:
        return f"{self.name}:\n\t{self.hand.__repr__()}\n"
