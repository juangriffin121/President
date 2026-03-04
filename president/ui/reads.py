from president.card import Card, Joker
from president.ui import writes


def prompt_indices(prompt: str, hand: list[Card | Joker]) -> list[int] | None:
    while True:
        writes.write(prompt)
        writes.print_hand(hand)
        writes.print_groups(hand)
        raw = input("Indices (space-separated), or empty to pass: ").strip()
        if raw == "":
            return []
        parts = raw.replace(",", " ").split()
        try:
            idxs = [int(x) for x in parts]
        except ValueError:
            writes.write(
                f"{writes.ANSI_RED}Invalid input. Use integers only.{writes.ANSI_RESET}"
            )
            continue
        if not idxs:
            return []
        if len(set(idxs)) != len(idxs):
            writes.write(
                f"{writes.ANSI_RED}Duplicate indices are not allowed.{writes.ANSI_RESET}"
            )
            continue
        if any(i < 0 or i >= len(hand) for i in idxs):
            writes.write(f"{writes.ANSI_RED}Index out of range.{writes.ANSI_RESET}")
            continue
        return idxs
