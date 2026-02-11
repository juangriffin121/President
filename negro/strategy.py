from abc import ABC, abstractmethod

from card import Card, Joker


class Strategy(ABC):
    @abstractmethod
    def choose_cards(self, global_state, player_state) -> list[Card | Joker] | None:
        pass


class Pass(Strategy):
    def choose_cards(self, global_state, player_state) -> list[Card | Joker] | None:
        return None
