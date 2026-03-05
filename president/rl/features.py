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
    rank_counts = [0] * 12
    for c in hand:
        if isinstance(c, Card):
            rank_counts[c.num - 1] += 1

    num_distinct_ranks = sum(1 for cnt in rank_counts if cnt > 0)
    num_pairs = sum(1 for cnt in rank_counts if cnt >= 2)
    num_triples = sum(1 for cnt in rank_counts if cnt >= 3)
    num_quads = sum(1 for cnt in rank_counts if cnt >= 4)

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

    state_features = [
        last_play_exists,
        last_play_rank_norm,
        last_play_count / 6,
        players_left / state.total_players,
        winners / state.total_players,
        is_president,
        is_vice_president,
        is_vice_scum,
        is_scum,
        bin_count / 50,
        bin_jokers / 2,
        bin_aces / 4,
        bin_avg_rank_norm,
        sum_other_hand_sizes / safe_other_total,
        avg_other_hand_size / initial_hand,
    ]

    return state_features


def get_action_features(
    action: list[Card | Joker] | None,
    hand_count: int,
    last_play: list[Card | Joker] | None,
    last_play_rank: int,
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

    action_features = [
        action_is_pass,
        action_count / 6,
        action_rank_norm,
        jokers_used / 2,
        uses_ace,
        empties_hand,
        fraction_hand_used,
        beats_last_by,
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

    action_feature_rows = []
    for action in valid_actions:
        action_feature_rows.append(
            get_action_features(action, hand_count, last_play, last_play_rank)
        )

    return Features(
        np.array(hand_features + state_features, dtype=float),
        np.array(action_feature_rows, dtype=float),
    )
