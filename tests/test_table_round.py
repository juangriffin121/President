from negro.card import Card
from negro.player import Player
from negro.state import GlobalState, PlayerState
from negro.strategy import Strategy
from negro.table import Table


class OpenThenPass(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card] | None:
        if not global_state.played:
            return [player_state.hand[0]]
        return None

    def choose_worst(self, count, hand):
        return hand[:count]

    def inform_of_results(self, performance: int, name: str):
        return None


class AlwaysPass(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card] | None:
        return None

    def choose_worst(self, count, hand):
        return hand[:count]

    def inform_of_results(self, performance: int, name: str):
        return None


class PlayAll(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card] | None:
        if not global_state.played:
            return [player_state.hand[0]]
        return None

    def choose_worst(self, count, hand):
        return hand[:count]

    def inform_of_results(self, performance: int, name: str):
        return None


def test_round_pass_cycle_clears_played_into_bin_and_returns_last_player() -> None:
    p1 = Player("p1", OpenThenPass())
    p2 = Player("p2", AlwaysPass())
    p3 = Player("p3", AlwaysPass())
    t = Table([p1, p2, p3])

    played_card = Card(3, "S")
    p1.hand = [played_card, Card(9, "S")]
    p2.hand = [Card(4, "S")]
    p3.hand = [Card(5, "S")]

    winner_idx = t.round(0)

    assert winner_idx == 0
    assert t.played == []
    assert t.bin == [played_card]
    assert len(p1.hand) == 1
    assert len(p2.hand) == 1
    assert len(p3.hand) == 1


def test_round_player_finishing_returns_next_player_index() -> None:
    p1 = Player("p1", PlayAll())
    p2 = Player("p2", AlwaysPass())
    p3 = Player("p3", AlwaysPass())
    t = Table([p1, p2, p3])

    winning_card = Card(7, "S")
    p1.hand = [winning_card]
    p2.hand = [Card(4, "S")]
    p3.hand = [Card(5, "S")]

    next_idx = t.round(0)

    assert next_idx == 0
    assert [p.id for p in t.players] == [1, 2]
    assert [w.id for w in t.winners] == [0]
    assert winning_card in t.bin


def test_round_all_passes_terminates_without_infinite_loop() -> None:
    p1 = Player("p1", AlwaysPass())
    p2 = Player("p2", AlwaysPass())
    p3 = Player("p3", AlwaysPass())
    t = Table([p1, p2, p3])

    p1.hand = [Card(3, "S")]
    p2.hand = [Card(4, "S")]
    p3.hand = [Card(5, "S")]

    winner_idx = t.round(0)

    assert winner_idx == 2
    assert t.played == []
    assert t.bin == []

