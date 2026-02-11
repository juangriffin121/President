from negro.card import Card, Joker


def order(card_num) -> int:
    if card_num == 1:
        return 13
    else:
        return card_num


def valid_choice(choice, last_choice) -> bool:
    result = True
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


def get_num(cards: list[Card | Joker]) -> int:
    num = None
    for card in cards:
        if type(card) == Card:
            num = card.num
            break
    if num is None:
        num = 1  # two jokers
    return num


def is_bigger(choice, last_choice) -> bool:
    num = get_num(choice)
    last_num = get_num(last_choice)
    return order(num) > order(last_num)
