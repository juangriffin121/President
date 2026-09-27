"""Logging for the "positional advantage" experiment.

Each row records, for one game played at one table, which persistent
seat (`Player.id`, assigned once when the `Table` is built and never
reassigned) held each role *during* that game, plus the finishing order
that game produced -- which is what determines next game's roles.

This is deliberately decoupled from the rest of the `president` package
(only stdlib is used) so it can be unit-tested and reused by any future
plotting code without dragging in the game engine.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GameLogEntry:
    trial_id: str
    game_index: int  # 1-based index of the game within the trial
    num_players: int
    president_id: int | None
    vice_president_id: int | None
    vice_scum_id: int | None
    scum_id: int | None
    winner_ids: list[int]
    hand_strength_prediction: list[float] 

    def to_row(self) -> dict:
        row = asdict(self)
        row["winner_ids"] = ",".join(str(w) for w in self.winner_ids)
        row["hand_strength_prediction"] = ",".join(str(s) for s in self.hand_strength_prediction)
        return row

    @staticmethod
    def field_names() -> list[str]:
        return [
            "trial_id",
            "game_index",
            "num_players",
            "president_id",
            "vice_president_id",
            "vice_scum_id",
            "scum_id",
            "winner_ids",  # <- was missing, add it or the column never gets written
            "hand_strength_prediction"
        ]

    @staticmethod
    def from_row(row: dict) -> "GameLogEntry":
        def _opt_int(v):
            return None if v in (None, "", "None") else int(v)

        winner_str = row.get("winner_ids") or ""
        winner_ids = [int(x) for x in winner_str.split(",") if x != ""]

        hand_strength_str = row.get("winner_ids") or ""
        hand_strength_prediction = [float(x) for x in hand_strength_str.split(",") if x != ""]

        return GameLogEntry(
            trial_id=row["trial_id"],
            game_index=int(row["game_index"]),
            num_players=int(row["num_players"]),
            president_id=_opt_int(row["president_id"]),
            vice_president_id=_opt_int(row["vice_president_id"]),
            vice_scum_id=_opt_int(row["vice_scum_id"]),
            scum_id=_opt_int(row["scum_id"]),
            winner_ids=winner_ids,
            hand_strength_prediction = hand_strength_prediction
        )


class GameLogWriter:
    """Append-only CSV writer that flushes after every row.

    Long experiments can run for hours; flushing eagerly means a crash
    (or a Ctrl-C) never loses more than the game in progress.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=GameLogEntry.field_names())
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def write(self, entry: GameLogEntry) -> None:
        self._writer.writerow(entry.to_row())
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "GameLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> list[GameLogEntry]:
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [GameLogEntry.from_row(row) for row in reader]


def merge_logs(paths: list[str | Path], output_path: str | Path) -> None:
    """Concatenate several log files into one (e.g. after running trials
    in parallel processes, each writing to its own file).
    """
    entries: list[GameLogEntry] = []
    for path in paths:
        entries.extend(read_log(path))
    entries.sort(key=lambda e: (e.trial_id, e.game_index))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GameLogEntry.field_names())
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_row())
