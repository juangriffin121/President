from abc import ABC, abstractmethod
from card import Card, Joker
from state import PlayerState
from rl_numpy import Agent
from ui import writes
from ranking import get_num, order_num, sort_hand
from state import GlobalState
from ui import reads
from utils import possible_sets


class Strategy(ABC):
    @abstractmethod
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        pass

    @abstractmethod
    def choose_worst(self, count, hand) -> list[Card | Joker]:
        # It should never be joker but the typing system makes things difficult
        pass

    @abstractmethod
    def inform_of_results(self, performance: int, name: str):
        pass


class Pass(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        return None

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        return _pick_min(count, hand)

    def inform_of_results(self, performance: int, name: str):
        match performance:
            case 2:
                writes.write(f"{name}: There's no way I won")
            case 1:
                writes.write(f"{name}: There's no way I,m second place")
            case 0:
                writes.write(f"{name}: This guys must be bad")
            case -1:
                writes.write(f"{name}: Expected")
            case -2:
                writes.write(f"{name}: That tracks")


class Smallest(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        played = global_state.played if global_state else []
        last_choice = played[-1] if played else None

        if last_choice is None:
            best = None
            for choice in possible_sets(player_state.hand):
                num = get_num(choice)
                key = order_num(num)
                if best is None or key < best[0]:
                    best = (key, choice)
            return None if best is None else best[1]

        last_num = get_num(last_choice)
        last_len = len(last_choice)
        best = None
        for choice in possible_sets(player_state.hand):
            if len(choice) != last_len:
                continue
            num = get_num(choice)
            if order_num(num) <= order_num(last_num):
                continue
            key = order_num(num)
            if best is None or key < best[0]:
                best = (key, choice)
        return None if best is None else best[1]

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        return _pick_min(count, hand)

    def inform_of_results(self, performance: int, name: str):
        match performance:
            case 2:
                writes.write(f"{name}: Yayyyyy")
            case 1:
                writes.write(f"{name}: Yay")
            case 0:
                writes.write(f"{name}: Meh")
            case -1:
                writes.write(f"{name}: :c")
            case -2:
                writes.write(f"{name}: :'c")


class UserStrategy(Strategy):
    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        played = global_state.played if global_state else []
        last_choice = played[-1] if played else None
        last_txt = f"{last_choice}" if last_choice is not None else "None"
        idxs = reads.prompt_indices(
            f"Choose cards to play. Last play: {last_txt}", player_state.hand
        )
        if not idxs:
            return None
        chosen = []
        for i in idxs:
            if 0 <= i < len(player_state.hand):
                chosen.append(player_state.hand[i])
        return chosen if chosen else None

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        idxs = reads.prompt_indices(f"Choose {count} cards to give.", hand)
        if idxs is None or len(idxs) != count:
            # fallback: smallest by rank, jokers highest
            non_jokers = [c for c in hand if not isinstance(c, Joker)]
            sorted_hand = sort_hand(non_jokers, joker_value=13)
            if len(sorted_hand) >= count:
                return sorted_hand[:count]
            remaining = count - len(sorted_hand)
            jokers = [c for c in hand if isinstance(c, Joker)]
            return sorted_hand + jokers[:remaining]
        chosen = []
        for i in idxs:
            if 0 <= i < len(hand):
                chosen.append(hand[i])
        if len(chosen) != count:
            # fallback
            non_jokers = [c for c in hand if not isinstance(c, Joker)]
            sorted_hand = sort_hand(non_jokers, joker_value=13)
            if len(sorted_hand) >= count:
                return sorted_hand[:count]
            remaining = count - len(sorted_hand)
            jokers = [c for c in hand if isinstance(c, Joker)]
            return sorted_hand + jokers[:remaining]
        return chosen

    def inform_of_results(self, performance: int, name: str):
        match performance:
            case 2:
                writes.write(f"{name}: You won, well done mister president")
            case 1:
                writes.write(f"{name}: Not bad")
            case 0:
                writes.write(f"{name}: You can do better than that")
            case -1:
                writes.write(f"{name}: Thats rough")
            case -2:
                writes.write(f"{name}: Haha loser")


class AgentStrategy(Strategy):
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.last_reward: int | None = None

    def choose_cards(
        self, global_state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        return self.agent.choose_cards(global_state, player_state)

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        return super().choose_worst(count, hand)

    def inform_of_results(self, performance: int, name: str):
        self.last_reward = performance
        if not self.agent.frozen:
            writes.write("I'll be learning from this")
            self.agent.update(performance)

        match performance:
            case 2:
                writes.write(f"{name}: AI will take over")
            case 1:
                writes.write(f"{name}: Not bad for a robot")
            case 0:
                writes.write(f"{name}: I'll get better")
            case -1:
                writes.write(f"{name}: I'm trying my best")
            case -2:
                writes.write(f"{name}: Useless clanker")


def _pick_min(count: int, hand: list[Card | Joker]) -> list[Card | Joker]:
    # keep jokers if possible
    non_jokers: list[Card | Joker] = [
        c for c in hand if not isinstance(c, Joker)
    ]  # typing system forces me to make that the type
    sorted_hand = sort_hand(non_jokers, joker_value=13)
    if len(sorted_hand) >= count:
        return sorted_hand[:count]
    remaining = count - len(sorted_hand)
    jokers = [c for c in hand if isinstance(c, Joker)]
    return sorted_hand + jokers[:remaining]
