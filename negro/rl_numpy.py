from numpy import ndarray
import numpy as np
from card import Card, Joker
from state import GlobalState, PlayerState
from rules import valid_choice
from utils import possible_sets
from ranking import get_num, order_num


class Agent:
    def __init__(self) -> None:
        self.trajectory = []
        self.worst_chosen: list[Card | Joker] = []
        self.frozen = False
        self.weights: ndarray | None = None
        self.dt = 0.6
        # self.baseline = 0.0
        # self.baseline_decay = 0.99
        self.temperature = 3.0

    def choose_cards(
        self, state: GlobalState, player_state
    ) -> list[Card | Joker] | None:
        last_played = state.played[-1] if state.played else None
        valid_actions = self.get_valid_actions(last_played, player_state.hand)
        features = self.get_features(state, player_state, valid_actions)
        if self.weights is None:
            self.weights = np.random.normal(0.0, 4, size=features.shape[1])
        probs = self.get_probabilities(features)
        choice_idx, choice = self.choose(valid_actions, probs)
        if not self.frozen:
            self.trajectory.append((features, choice_idx, probs))
        return choice

    def choose_worst(self, count, hand):
        # code
        if not self.frozen:
            pass
            # self.worst_chosen =

    def update(self, reward):
        assert not self.frozen
        # baseline seems not to affect much
        # self.baseline = (
        #     self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        # )
        advantage = reward  # - self.baseline
        for features, choice_idx, probs in self.trajectory:
            one_hot = np.zeros(probs.shape[0])
            one_hot[choice_idx] = 1
            grad = advantage * (one_hot - probs)
            self.weights += self.dt * grad @ features
        self.trajectory = []

    def get_probabilities(self, features: ndarray) -> ndarray:
        assert self.weights is not None
        if features.size == 0:  # No possible actions
            return np.zeros(features.shape[0], dtype=float)
        scores = (features @ self.weights) / max(self.temperature, 1e-6)
        max_score = np.max(scores) if scores.size else 0
        exp_scores = np.exp(scores - max_score)
        total = exp_scores.sum()
        if total <= 0:
            return np.zeros_like(exp_scores)
        return exp_scores / total

    def choose(
        self, actions: list[list[Card | Joker] | None], probs: ndarray
    ) -> tuple[int | None, list[Card | Joker] | None]:
        if len(actions) == 0:
            return (None, None)
        if len(actions) == 1:
            return (0, actions[0])
        assert probs.size == len((actions))
        total = probs.sum()
        if total <= 0:
            idx = int(np.random.randint(0, len(actions)))
            return (idx, actions[idx])
        probs = probs / total
        idx = int(np.random.choice(len(actions), p=probs))
        return (idx, actions[idx])

    def get_valid_actions(
        self, last_played: list[Card | Joker] | None, hand: list[Card | Joker]
    ) -> list[list[Card | Joker] | None]:
        possible: list[list[Card | Joker] | None] = list(possible_sets(hand))
        if last_played is None:
            return possible
        valid = [choice for choice in possible if valid_choice(choice, last_played)]
        valid.append(None)
        return valid

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_features(
        self,
        state: GlobalState,
        player_state: PlayerState,
        valid_actions: list[list[Card | Joker] | None],
    ) -> ndarray:
        hand = player_state.hand
        player_id = player_state.id
        hand_count = len(hand)
        initial_hand = 50 / state.total_players
        safe_hand_count = max(1, hand_count)
        safe_other_total = max(1, 50 - initial_hand)

        jokers_in_hand = sum(1 for c in hand if isinstance(c, Joker))
        rank_counts = [0] * 12
        for c in hand:
            if isinstance(c, Card):
                rank_counts[c.num - 1] += 1

        num_distinct_ranks = sum(1 for cnt in rank_counts if cnt > 0)
        num_pairs = sum(1 for cnt in rank_counts if cnt >= 2)
        num_triples = sum(1 for cnt in rank_counts if cnt >= 3)
        num_quads = sum(1 for cnt in rank_counts if cnt >= 4)

        last_play_exists = 1 if state.played else 0
        last_play = state.played[-1] if state.played else None
        last_play_rank = get_num(last_play) if last_play else 0
        last_play_rank_norm = order_num(last_play_rank) / 13 if last_play else 0
        last_play_count = len(last_play) if last_play else 0

        players_left = state.total_players - len(state.winners)
        winners = len(state.winners)

        is_presidente = (
            int(player_id == state.presidente) if state.presidente is not None else 0
        )
        is_vice_presidente = (
            int(player_id == state.vice_presidente)
            if state.vice_presidente is not None
            else 0
        )
        is_vice_negro = (
            int(player_id == state.vice_negro) if state.vice_negro is not None else 0
        )
        is_negro = int(player_id == state.negro) if state.negro is not None else 0

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

        sum_other_hand_sizes = sum(state.hand_sizes) - hand_count
        avg_other_hand_size = (
            sum_other_hand_sizes / max(1, players_left - 1) if players_left > 1 else 0
        )

        state_features = [
            hand_count / initial_hand,
            jokers_in_hand / 2,
            *[cnt / 4 for cnt in rank_counts],
            num_distinct_ranks / 12,
            num_pairs / (initial_hand / 2),
            num_triples / (initial_hand / 3),
            num_quads / (initial_hand / 4),
            last_play_exists,
            last_play_rank_norm,
            last_play_count / 6,
            players_left / state.total_players,
            winners / state.total_players,
            is_presidente,
            is_vice_presidente,
            is_vice_negro,
            is_negro,
            bin_count / 50,
            bin_jokers / 2,
            bin_aces / 4,
            bin_avg_rank_norm,
            sum_other_hand_sizes / safe_other_total,
            avg_other_hand_size / initial_hand,
        ]

        feature_rows = []
        for action in valid_actions:
            action_is_pass = 1 if action is None else 0
            action_count = 0 if action is None else len(action)
            action_rank = 0 if action is None else order_num(get_num(action))
            action_rank_norm = action_rank / 13 if action is not None else 0
            jokers_used = (
                sum(1 for c in action if isinstance(c, Joker)) if action else 0
            )
            uses_ace = (
                1
                if action is not None
                and any(isinstance(c, Card) and c.num == 1 for c in action)
                else 0
            )
            empties_hand = 1 if action and action_count == hand_count else 0
            fraction_hand_used = (
                action_count / safe_hand_count if action is not None else 0
            )
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

            feature_rows.append(state_features + action_features)

        return np.array(feature_rows, dtype=float)
