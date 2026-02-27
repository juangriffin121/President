from itertools import combinations
from typing import Iterator

from negro.card import Card, Joker
from negro.ranking import order_num


def possible_sets(hand: list[Card | Joker]) -> Iterator[list[Card | Joker]]:
    cards = [c for c in hand if not isinstance(c, Joker)]
    jokers = [c for c in hand if isinstance(c, Joker)]
    nums = sorted(set(c.num for c in cards), key=order_num)

    for n in range(1, len(hand) + 1):
        for num in nums:
            matching = [c for c in cards if c.num == num]
            for k in range(min(len(jokers), n) + 1):
                needed = n - k
                if needed <= len(matching):
                    for combo in combinations(matching, needed):
                        yield list(combo) + jokers[:k]
    for n in range(1, len(jokers) + 1):
        yield list(jokers[:n])
