from president.player import Player
from president.strategy import Smallest
from president.table import Table


def test_table_game_terminates_and_collects_all_cards() -> None:
    players = [
        Player("p1", Smallest()),
        Player("p2", Smallest()),
        Player("p3", Smallest()),
    ]
    table = Table(players)

    table.game()

    assert len(table.winners) == 3
    assert len(table.players) == 0
    assert len(table.deck.cards) == 50
    assert all(len(p.hand) == 0 for p in table.winners)
    assert sorted(p.id for p in table.winners) == [0, 1, 2]
