from dataclasses import dataclass

from negro.card import Card, Joker


@dataclass
class GlobalState:
    played: list[list[Card | Joker]]
    bin: list[Card | Joker]
    players: list[int]
    hand_sizes: list[int]
    winners: list[int]
    total_players: int
    presidente: int | None
    vice_presidente: int | None
    vice_negro: int | None
    negro: int | None


@dataclass
class PlayerState:
    id: int
    hand: list[Card | Joker]
