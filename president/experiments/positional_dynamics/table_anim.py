"""Render / animate the president-scum roles rotating around the table
across many games.

`draw_table` is the single primitive both the static single-frame plot
and the animation build on, so there's one place that knows how to draw
a table snapshot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from president.experiments.positional_dynamics.logger import GameLogEntry, read_log
from president.experiments.positional_dynamics.table_viz import EDGE_COLOR, all_seat_styles, seat_positions

NODE_RADIUS = 0.12


def draw_table(
    ax: plt.Axes,
    num_players: int,
    president_id: int | None,
    vice_president_id: int | None,
    vice_scum_id: int | None,
    scum_id: int | None,
    show_seat_ids: bool = True,
) -> list[Circle]:
    """Draw one static snapshot of the table onto `ax` (cleared first).

    Returns the Circle patches, in case a caller wants to restyle them
    directly instead of clearing and redrawing (not currently used, but
    kept for anyone building a faster/blitted animation later).
    """
    ax.clear()
    positions = seat_positions(num_players)
    styles = all_seat_styles(
        num_players, president_id, vice_president_id, vice_scum_id, scum_id
    )

    circles = []
    for seat, ((x, y), style) in enumerate(zip(positions, styles)):
        circle = Circle(
            (x, y),
            NODE_RADIUS,
            facecolor=style.color,
            alpha=style.alpha,
            edgecolor=EDGE_COLOR,
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        circles.append(circle)
        if show_seat_ids:
            ax.text(x, y, str(seat), ha="center", va="center", fontsize=8, zorder=4)

    # Draw arrows showing the direction of the round
    for i in range(num_players):
        ax.annotate(
            "",
            xy=positions[(i + 1) % num_players],
            xytext=positions[i],
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.5,
                color=EDGE_COLOR,
                shrinkA=12,
                shrinkB=12,
            ),
            zorder=1,
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return circles


def plot_single_game(
    entry: GameLogEntry,
    ax: plt.Axes | None = None,
    show_seat_ids: bool = True,
) -> plt.Axes:
    """Plot a single game's role snapshot. Handy for spot-checking a log
    or for building your own custom grid of snapshots."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    assert(ax is not None)
    draw_table(
        ax,
        entry.num_players,
        entry.president_id,
        entry.vice_president_id,
        entry.vice_scum_id,
        entry.scum_id,
        show_seat_ids=show_seat_ids,
    )
    ax.set_title(f"Game {entry.game_index} (trial {entry.trial_id})")
    return ax


def animate_trial(
    log_path: str | Path,
    trial_id: str,
    interval_ms: int = 120,
    show_seat_ids: bool = True,
    game_range: tuple[int, int] | None = None,
    debug: bool = False
) -> animation.FuncAnimation:
    """Build an animation of role assignments, game by game, for a
    single trial.

    `game_range` (inclusive, 1-based `(first, last)`) lets you animate a
    slice of a long trial instead of the whole thing.

    Call `.save(...)` on the result, or `plt.show()` it (e.g. in a
    notebook) -- see `save_animation` for a thin wrapper.
    """
    entries = [e for e in read_log(log_path) if e.trial_id == trial_id]
    entries.sort(key=lambda e: e.game_index)
    if game_range is not None:
        first, last = game_range
        entries = [e for e in entries if first <= e.game_index <= last]
    if not entries:
        raise ValueError(f"No games found for trial_id={trial_id} in {log_path}")

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_idx: int):
        entry = entries[frame_idx]
        if debug:
            if frame_idx%10 == 0:
                print(f"Frame: {frame_idx}")
        draw_table(
            ax,
            entry.num_players,
            entry.president_id,
            entry.vice_president_id,
            entry.vice_scum_id,
            entry.scum_id,
            show_seat_ids=show_seat_ids,
        )
        ax.set_title(f"Game {entry.game_index} (trial {entry.trial_id})")
        return ax.patches

    anim = animation.FuncAnimation(
        fig, update, frames=len(entries), interval=interval_ms, blit=False
    )
    return anim


def save_animation(anim: animation.FuncAnimation, path: str | Path, fps: int = 5) -> None:
    """Save to .gif (via ffmpeg, no external deps) or .mp4 (needs ffmpeg
    on PATH), based on the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        anim.save(str(path), writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(str(path), writer="ffmpeg", fps=fps)
    plt.close(anim._fig)
