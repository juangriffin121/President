"""
Load a President simulation's GameLogEntry CSV into pandas dataframes
ready for analysis.

load_game_log(path)   -> one row per game, with proper dtypes and
                          winner_ids parsed back into a real list.

to_player_frame(games) -> one row per (trial, game, player), with:
                          - seat_offset: this player's seat relative to
                            the president *for this game* (fixed frame,
                            since player_id never changes within a trial)
                          - role_this_game: president/vice_president/
                            vice_scum/scum/commoner, i.e. the role they
                            held going into this game
                          - finish_rank: 0-based position in winner_ids
                            (0 = finished first)
                          - performance: the +2/+1/0/-1/-2 grade Table.game()
                            computes from finish_rank, i.e. the role this
                            player earns for the *next* game
"""

import pandas as pd

def load_game_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Nullable ints: first game of a trial has no assigned roles yet, and
    # 3-player tables never get vice roles
    for col in ["president_id", "vice_president_id", "vice_scum_id", "scum_id"]:
        df[col] = df[col].astype("Int64")

    def _parse_winners(s):
        if pd.isna(s) or s == "":
            return []
        return [int(x) for x in str(s).split(",")]

    def _parse_strengths(s):
        if pd.isna(s) or s == "":
            return []
        return [float(x) for x in str(s).split(",")]

    df["winner_ids"] = df["winner_ids"].apply(_parse_winners)
    df["hand_strength_prediction"] = df["hand_strength_prediction"].apply(_parse_strengths)

    return df.sort_values(["trial_id", "game_index"]).reset_index(drop=True)


def _performance(finish_rank: int, n: int) -> int:
    """Mirrors the grading logic at the bottom of Table.game()."""
    vices = (1 != n - 2)  # vice_president_idx != vice_scum_idx -> False only when n == 3
    if finish_rank == 0:
        return 2
    if finish_rank == n - 1:
        return -2
    if vices and finish_rank == 1:
        return 1
    if vices and finish_rank == n - 2:
        return -1
    return 0


def to_player_frame(games: pd.DataFrame) -> pd.DataFrame:
    role_cols = {
        "president_id": "president",
        "vice_president_id": "vice_president",
        "vice_scum_id": "vice_scum",
        "scum_id": "scum",
    }

    rows = []
    for row in games.itertuples(index=False):
        n = row.num_players
        pres = row.president_id

        role_by_id = {}
        for col, label in role_cols.items():
            pid = getattr(row, col)
            if pd.notna(pid):
                role_by_id[int(pid)] = label

        rank_by_id = {pid: rank for rank, pid in enumerate(row.winner_ids)}
        hand_strength_by_id = {pid: hand_strength for hand_strength, pid in zip(row.hand_strength_prediction, row.winner_ids)}

        for pid in range(n):
            finish_rank = rank_by_id.get(pid)
            rows.append({
                "trial_id": row.trial_id,
                "game_index": row.game_index,
                "num_players": n,
                "player_id": pid,
                "seat_offset": (pid - pres) % n if pd.notna(pres) else pd.NA,
                "role_this_game": role_by_id.get(pid, "commoner"),
                "finish_rank": finish_rank if finish_rank is not None else pd.NA,
                "performance": _performance(finish_rank, n) if finish_rank is not None else pd.NA,
                "hand_strength_prediction": hand_strength_by_id.get(pid)
            })

    players = pd.DataFrame(rows)
    for col in ["seat_offset", "finish_rank", "performance"]:
        players[col] = players[col].astype("Int64")
    return players


if __name__ == "__main__":
    games = load_game_log("positional_dynamics_run4")
    players = to_player_frame(games)
    print(games.head())
    print(players.head())
