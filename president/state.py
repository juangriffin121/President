from dataclasses import dataclass

from president.card import Card, Joker


@dataclass
class GlobalState:
    played: list[list[Card | Joker]]
    bin: list[Card | Joker]
    players: list[int]
    hand_sizes: list[int]
    winners: list[int]
    total_players: int
    president: int | None
    vice_president: int | None
    vice_scum: int | None
    scum: int | None
    last_played_by: int | None = None


@dataclass
class PlayerState:
    id: int
    hand: list[Card | Joker]
