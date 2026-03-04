from president.card import Card, Joker
from president.utils import possible_sets


def test_possible_sets_include_expected_joker_combinations() -> None:
    hand = [Card(5, "A"), Card(5, "B"), Joker()]
    sets = list(possible_sets(hand))

    assert sets
    assert all(len(choice) > 0 for choice in sets)

    assert any(
        len(choice) == 3
        and sum(isinstance(c, Joker) for c in choice) == 1
        and sum(isinstance(c, Card) and c.num == 5 for c in choice) == 2
        for choice in sets
    )
    assert any(len(choice) == 1 and isinstance(choice[0], Joker) for choice in sets)


def test_possible_sets_without_jokers_only_yields_existing_ranks() -> None:
    hand = [Card(2, "A"), Card(3, "B")]
    sets = list(possible_sets(hand))

    assert len(sets) == 2
    assert all(len(choice) == 1 for choice in sets)
    assert sorted(choice[0].num for choice in sets if isinstance(choice[0], Card)) == [
        2,
        3,
    ]

