from dataclasses import dataclass
import numpy as np

from president.card import Card, Joker
from president.ranking import get_num, order_num
from president.state import GlobalState, PlayerState


@dataclass
class Features:
    state_hand: np.ndarray  # shape (F_s,)
    actions: np.ndarray  # shape (A, F_a)

    def as_concatenated(self) -> np.ndarray:
        # for old agents
        if self.actions.shape[0] == 0:
            return np.zeros((0, self.state_hand.shape[0] + self.actions.shape[1]))
        s = np.repeat(self.state_hand[None, :], self.actions.shape[0], axis=0)
        return np.concatenate([s, self.actions], axis=1)


def get_hand_features(hand: list[Card | Joker], total_players: int) -> list[float]:
    hand_count = len(hand)
    initial_hand = 50 / total_players

    jokers_in_hand = sum(1 for c in hand if isinstance(c, Joker))
    rank_counts = _rank_counts(hand)
    num_distinct_ranks, num_pairs, num_triples, num_quads = _combo_counts(rank_counts)

    feats = [
        hand_count / initial_hand,
        jokers_in_hand / 2,
        *[cnt / 4 for cnt in rank_counts],
        num_distinct_ranks / 12,
        num_pairs / (initial_hand / 2),
        num_triples / (initial_hand / 3),
        num_quads / (initial_hand / 4),
    ]
    return feats


def _rank_counts(cards: list[Card | Joker]) -> list[int]:
    counts = [0] * 12
    for c in cards:
        if isinstance(c, Card):
            counts[c.num - 1] += 1
    return counts


def _combo_counts(rank_counts: list[int]) -> tuple[int, int, int, int]:
    distinct = sum(1 for cnt in rank_counts if cnt > 0)
    pairs = sum(1 for cnt in rank_counts if cnt >= 2)
    triples = sum(1 for cnt in rank_counts if cnt >= 3)
    quads = sum(1 for cnt in rank_counts if cnt >= 4)
    return distinct, pairs, triples, quads


def _remaining_hand_after_action(
    hand: list[Card | Joker], action: list[Card | Joker] | None
) -> list[Card | Joker]:
    if action is None:
        return hand.copy()

    remaining = hand.copy()
    for c in action:
        if c in remaining:
            remaining.remove(c)
    return remaining


def _known_rank_and_joker_counts(
    state: GlobalState,  # hand: list[Card | Joker]
) -> tuple[list[int], int]:
    # how many of each card where played
    known_cards: list[Card | Joker] = (
        []
    )  # hand.copy() counting or not counting hand held cards?
    known_cards.extend(state.bin)
    known_cards.extend(sum(state.played, []))

    rank_counts = _rank_counts(known_cards)
    joker_count = sum(1 for c in known_cards if isinstance(c, Joker))
    return rank_counts, joker_count


def get_state_features(
    state: GlobalState, player_id: int, hand_count: int
) -> list[float]:
    initial_hand = 50 / state.total_players
    safe_other_total = max(1, 50 - initial_hand)
    sum_other_hand_sizes = sum(state.hand_sizes) - hand_count

    last_play_exists = 1 if state.played else 0
    last_play = state.played[-1] if state.played else None
    last_play_rank = get_num(last_play) if last_play else 0
    last_play_rank_norm = order_num(last_play_rank) / 13 if last_play else 0
    last_play_count = len(last_play) if last_play else 0

    players_left = state.total_players - len(state.winners)
    winners = len(state.winners)

    is_president = (
        int(player_id == state.president) if state.president is not None else 0
    )
    is_vice_president = (
        int(player_id == state.vice_president)
        if state.vice_president is not None
        else 0
    )
    is_vice_scum = (
        int(player_id == state.vice_scum) if state.vice_scum is not None else 0
    )
    is_scum = int(player_id == state.scum) if state.scum is not None else 0

    bin_cards = state.bin
    bin_count = len(bin_cards)
    bin_jokers = sum(1 for c in bin_cards if isinstance(c, Joker))
    bin_aces = sum(1 for c in bin_cards if isinstance(c, Card) and c.num == 1)
    if bin_count > 0:
        bin_avg_rank_norm = (
            sum(order_num(c.num) for c in bin_cards if isinstance(c, Card))
            / max(1, bin_count - bin_jokers)
        ) / 13
    else:
        bin_avg_rank_norm = 0

    avg_other_hand_size = (
        sum_other_hand_sizes / max(1, players_left - 1) if players_left > 1 else 0
    )

    # New turn-order context features.
    active_players = state.players
    player_idx = active_players.index(player_id) if player_id in active_players else 0
    current_seat_norm = player_idx / max(1, players_left - 1)

    last_played_by = state.last_played_by
    if last_played_by is not None:
        assert (
            last_played_by in active_players
        )  # If a play was a winning move it gets sent to the bin and the next in line plays
        assert last_played_by != player_id  # Agent should have won in that case
        assert players_left > 1  # Agent would have lost by then
        last_player_idx = active_players.index(last_played_by)
        seats_from_last_play = (player_idx - last_player_idx) % players_left
        seats_from_last_play_norm = seats_from_last_play / max(1, players_left - 1)
    else:
        seats_from_last_play_norm = 0

    # New opponent pressure features.
    if player_id in active_players:
        my_state_idx = active_players.index(player_id)
        other_hand_sizes = [
            hs for idx, hs in enumerate(state.hand_sizes) if idx != my_state_idx
        ]
    else:
        other_hand_sizes = state.hand_sizes.copy()
    min_other_hand_size = min(other_hand_sizes) if other_hand_sizes else 0
    opp_count = max(1, players_left - 1)
    opp_at_most_2 = sum(1 for hs in other_hand_sizes if hs <= 2) / opp_count
    opp_at_most_1 = sum(1 for hs in other_hand_sizes if hs <= 1) / opp_count

    state_features = [
        last_play_exists,
        last_play_rank_norm,
        last_play_count / 6,
        players_left / state.total_players,
        winners / state.total_players,
        # is_president,
        # is_vice_president,
        # is_vice_scum,
        # is_scum,
        bin_count / 50,
        bin_jokers / 2,
        bin_aces / 4,
        bin_avg_rank_norm,
        sum_other_hand_sizes / safe_other_total,
        avg_other_hand_size / initial_hand,
        current_seat_norm,
        seats_from_last_play_norm,
        min_other_hand_size / initial_hand,
        opp_at_most_2,
        opp_at_most_1,
    ]

    return state_features


def get_action_features(
    action: list[Card | Joker] | None,
    hand: list[Card | Joker],
    hand_count: int,
    last_play: list[Card | Joker] | None,
    last_play_rank: int,
    seen_rank_counts: list[int],
    seen_jokers: int,
) -> list[float]:
    safe_hand_count = max(1, hand_count)

    action_is_pass = 1 if action is None else 0
    action_count = 0 if action is None else len(action)
    action_rank = 0 if action is None else order_num(get_num(action))
    action_rank_norm = action_rank / 13 if action is not None else 0
    jokers_used = sum(1 for c in action if isinstance(c, Joker)) if action else 0
    uses_ace = (
        1
        if action is not None
        and any(isinstance(c, Card) and c.num == 1 for c in action)
        else 0
    )
    empties_hand = 1 if action and action_count == hand_count else 0
    fraction_hand_used = action_count / safe_hand_count if action is not None else 0
    beats_last_by = (
        (action_rank - order_num(last_play_rank)) / 13
        if action is not None and last_play
        else 0
    )

    # New action-cost features: whether this move breaks larger combos. [[red]]
    if action is None:
        breaks_larger_set = 0
        uses_joker_when_not_needed = 0
    else:
        has_non_joker = any(isinstance(c, Card) for c in action)
        if has_non_joker:
            rank = get_num(action)
            rank_in_hand = sum(1 for c in hand if isinstance(c, Card) and c.num == rank)
            breaks_larger_set = int(action_count < rank_in_hand)
            uses_joker_when_not_needed = int(
                jokers_used > 0 and rank_in_hand >= action_count
            )
        else:
            breaks_larger_set = 0
            uses_joker_when_not_needed = 0

    # New post-action hand-quality features.
    remaining_hand = _remaining_hand_after_action(hand, action)
    remaining_hand_count = len(remaining_hand)
    remaining_jokers = sum(1 for c in remaining_hand if isinstance(c, Joker))
    remaining_rank_counts = _rank_counts(remaining_hand)
    rem_distinct, rem_pairs, rem_triples, rem_quads = _combo_counts(
        remaining_rank_counts
    )
    remaining_high_cards = sum(
        1
        for c in remaining_hand
        if isinstance(c, Joker) or (isinstance(c, Card) and c.num == 1)
    )

    # New card-availability estimate: how many higher same-size responses remain unseen.
    if action is None:
        unseen_higher_group_ratio = 0
    else:
        response_size = max(1, action_count)
        action_rank_raw = get_num(action)
        action_order = order_num(action_rank_raw)

        unseen_groups = 0
        max_groups = 0
        for rank in range(1, 13):
            if order_num(rank) <= action_order:
                continue
            unseen_rank_cards = max(0, 4 - seen_rank_counts[rank - 1])
            unseen_groups += unseen_rank_cards // response_size
            max_groups += 4 // response_size

        unseen_joker_cards = max(0, 2 - seen_jokers)
        unseen_groups += unseen_joker_cards // response_size
        max_groups += 2 // response_size

        unseen_higher_group_ratio = unseen_groups / max(1, max_groups)

    action_features = [
        action_is_pass,
        action_count / 6,
        action_rank_norm,
        jokers_used / 2,
        uses_ace,
        empties_hand,
        fraction_hand_used,
        beats_last_by,
        breaks_larger_set,
        uses_joker_when_not_needed,
        remaining_hand_count / safe_hand_count,
        remaining_jokers / 2,
        rem_distinct / 12,
        rem_pairs / 12,
        rem_triples / 12,
        rem_quads / 12,
        remaining_high_cards / 6,
        unseen_higher_group_ratio,
    ]

    return action_features


def get_features(
    state: GlobalState,
    player_state: PlayerState,
    valid_actions: list[list[Card | Joker] | None],
) -> Features:
    hand = player_state.hand
    hand_count = len(player_state.hand)
    last_play = state.played[-1] if state.played else None
    last_play_rank = get_num(last_play) if last_play else 0

    hand_features = get_hand_features(hand, state.total_players)
    state_features = get_state_features(state, player_state.id, hand_count)
    seen_rank_counts, seen_jokers = _known_rank_and_joker_counts(state)

    action_feature_rows = []
    for action in valid_actions:

        action_feature_rows.append(
            get_action_features(
                action,
                hand,
                hand_count,
                last_play,
                last_play_rank,
                seen_rank_counts,
                seen_jokers,
            )
        )

    return Features(
        np.array(hand_features + state_features, dtype=float),
        np.array(action_feature_rows, dtype=float),
    )


def hand_feat_names():
    return [
        "hand_count / initial_hand",
        "jokers_in_hand / 2",
        *[f"{i}s_count" for i in range(1, 13)],
        "num_distinct_ranks / 12",
        "num_pairs / (initial_hand / 2)",
        "num_triples / (initial_hand / 3)",
        "num_quads / (initial_hand / 4)",
    ]


def state_feat_names():
    return [
        "last_play_exists",
        "last_play_rank_norm",
        "last_play_count / 6",
        "players_left / state.total_players",
        "winners / state.total_players",
        # is_president,
        # is_vice_president,
        # is_vice_scum,
        # is_scum,
        "bin_count / 50",
        "bin_jokers / 2",
        "bin_aces / 4",
        "bin_avg_rank_norm",
        "sum_other_hand_sizes / safe_other_total",
        "avg_other_hand_size / initial_hand",
        "current_seat_norm",
        "seats_from_last_play_norm",
        "min_other_hand_size / initial_hand",
        "opp_at_most_2",
        "opp_at_most_1",
    ]


def action_feat_names():
    return [
        "action_is_pass",
        "action_count / 6",
        "action_rank_norm",
        "jokers_used / 2",
        "uses_ace",
        "empties_hand",
        "fraction_hand_used",
        "beats_last_by",
        "breaks_larger_set",
        "uses_joker_when_not_needed",
        "remaining_hand_count / safe_hand_count",
        "remaining_jokers / 2",
        "rem_distinct / 12",
        "rem_pairs / 12",
        "rem_triples / 12",
        "rem_quads / 12",
        "remaining_high_cards / 6",
        "unseen_higher_group_ratio",
    ]
