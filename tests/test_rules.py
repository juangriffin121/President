from president.card import Card, Joker
from president.rules import all_cards_same, valid_choice


def test_all_cards_same_allows_jokers() -> None:
    cards = [Card(7, "A"), Joker(), Card(7, "B")]
    assert all_cards_same(cards)


def test_all_cards_same_rejects_mixed_numbers() -> None:
    cards = [Card(7, "A"), Joker(), Card(8, "B")]
    assert not all_cards_same(cards)


def test_valid_choice_pass_allowed_only_after_previous_play() -> None:
    assert not valid_choice(None, None)
    assert valid_choice(None, [Card(4, "A")])


def test_valid_choice_requires_same_size_and_bigger_rank() -> None:
    last_choice = [Card(5, "A"), Card(5, "B")]

    bigger_same_len = [Card(6, "A"), Joker()]
    wrong_len = [Card(6, "A")]
    not_bigger = [Card(4, "A"), Joker()]

    assert valid_choice(bigger_same_len, last_choice)
    assert not valid_choice(wrong_len, last_choice)
    assert not valid_choice(not_bigger, last_choice)

