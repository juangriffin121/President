"""Geometry and colour mapping for drawing the table as a regular
polygon with one node per seat.

Kept separate from `table_animation.py` so the pure math/styling
(easy to unit test, no matplotlib needed) is independent of the
rendering code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRESIDENT_COLOR = "#2ca02c"  # green
SCUM_COLOR = "#d62728"  # red
NEUTRAL_COLOR = "#f2f2f2"  # white-ish (pure white is invisible on a white bg)
EDGE_COLOR = "#333333"

MAIN_ALPHA = 1.0
VICE_ALPHA = 0.45
NEUTRAL_ALPHA = 0.9


@dataclass(frozen=True)
class SeatStyle:
    color: str
    alpha: float


def seat_positions(num_players: int, radius: float = 1.0) -> np.ndarray:
    """Vertices of a regular `num_players`-gon, shape (num_players, 2).

    Seat 0 is placed at the top, seats increase clockwise -- purely a
    display convention, it has no bearing on turn order.
    """
    if num_players < 1:
        raise ValueError("num_players must be >= 1")
    angles = np.pi / 2 - 2 * np.pi * np.arange(num_players) / num_players
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def seat_style(
    seat_id: int,
    president_id: int | None,
    vice_president_id: int | None,
    vice_scum_id: int | None,
    scum_id: int | None,
) -> SeatStyle:
    if president_id is not None and seat_id == president_id:
        return SeatStyle(PRESIDENT_COLOR, MAIN_ALPHA)
    if scum_id is not None and seat_id == scum_id:
        return SeatStyle(SCUM_COLOR, MAIN_ALPHA)
    if vice_president_id is not None and seat_id == vice_president_id:
        return SeatStyle(PRESIDENT_COLOR, VICE_ALPHA)
    if vice_scum_id is not None and seat_id == vice_scum_id:
        return SeatStyle(SCUM_COLOR, VICE_ALPHA)
    return SeatStyle(NEUTRAL_COLOR, NEUTRAL_ALPHA)


def all_seat_styles(
    num_players: int,
    president_id: int | None,
    vice_president_id: int | None,
    vice_scum_id: int | None,
    scum_id: int | None,
) -> list[SeatStyle]:
    return [
        seat_style(seat, president_id, vice_president_id, vice_scum_id, scum_id)
        for seat in range(num_players)
    ]
