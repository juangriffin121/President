from negro.card import Card, Joker
from negro.ranking import get_num, order_num


def valid_choice(
    choice: list[Card | Joker] | None, last_choice: list[Card | Joker] | None
) -> bool:
    result = True
    if choice is None:
        allowed: bool = last_choice is not None
        return allowed
    if last_choice is None:
        return True
    assert all_cards_same(last_choice)
    assert len(last_choice) >= 1
    result = result and all_cards_same(choice)
    result = result and is_bigger(choice, last_choice)
    result = result and len(choice) == len(last_choice)
    return result


def all_cards_same(cards: list[Card | Joker]) -> bool:
    result = True
    num = None
    for card in cards:
        if type(card) == Card:
            if num is None:
                num = card.num
            else:
                if card.num != num:
                    result = False
                    break
    return result


def is_bigger(choice, last_choice) -> bool:
    num = get_num(choice, joker_only_value=1)
    last_num = get_num(last_choice, joker_only_value=1)
    return order_num(num) > order_num(last_num)
