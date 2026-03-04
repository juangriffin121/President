from president.card import Card, Joker

HIGHEST_RANK = 13


def order_num(num: int) -> int:
    return HIGHEST_RANK if num == 1 else num


def get_num(cards: list[Card | Joker], joker_only_value: int = 1) -> int:
    for card in cards:
        if isinstance(card, Card):
            return card.num
    return joker_only_value


def rank_card(card: Card | Joker, joker_value: int = HIGHEST_RANK) -> int:
    if isinstance(card, Joker):
        return joker_value
    return order_num(card.num)


def sort_hand(
    hand: list[Card | Joker], joker_value: int = HIGHEST_RANK, reverse: bool = False
):
    return sorted(hand, key=lambda c: rank_card(c, joker_value), reverse=reverse)
