from negro.card import Card, Joker
from negro.ranking import get_num, order_num, rank_card, sort_hand


def test_order_num_treats_ace_as_highest() -> None:
    assert order_num(1) == 13
    assert order_num(12) == 12


def test_get_num_uses_default_for_joker_only_play() -> None:
    assert get_num([Joker(), Joker()], joker_only_value=7) == 7


def test_rank_card_respects_joker_value_override() -> None:
    assert rank_card(Joker()) == 13
    assert rank_card(Joker(), joker_value=0) == 0


def test_sort_hand_with_custom_joker_value() -> None:
    hand = [Card(3, "S"), Joker(), Card(1, "C")]
    sorted_hand = sort_hand(hand, joker_value=0, reverse=False)

    assert isinstance(sorted_hand[0], Joker)
    assert isinstance(sorted_hand[1], Card) and sorted_hand[1].num == 3
    assert isinstance(sorted_hand[2], Card) and sorted_hand[2].num == 1

