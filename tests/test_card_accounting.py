import random

import numpy as np

from president.player import Player
from president.strategy import Smallest
from president.table import Table


def _run_one_game(num_players: int, seed: int) -> Table:
    random.seed(seed)
    np.random.seed(seed)
    players = [Player(f"p{i}", Smallest()) for i in range(num_players)]
    table = Table(players)
    table.game()
    return table


def test_total_cards_preserved_after_game_for_various_player_counts() -> None:
    for num_players in (3, 4, 5):
        for seed in (1, 7, 19):
            table = _run_one_game(num_players=num_players, seed=seed)
            assert len(table.deck.cards) == 50
            assert len(table.bin) == 0
            assert len(table.players) == 0
            assert len(table.winners) == num_players
            assert all(len(p.hand) == 0 for p in table.winners)

