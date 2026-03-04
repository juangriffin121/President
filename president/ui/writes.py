from president.card import Card, Joker

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"

SILENT = False


def set_silent(value: bool) -> None:
    global SILENT
    SILENT = value


def write(message: str = "") -> None:
    if SILENT:
        return
    print(message)


def announce_turn(player_name: str) -> None:
    write(player_name)


def announce_play(played) -> None:
    write(str(played))


def announce_round_winner(player_idx: int) -> None:
    write(f"round won by player{player_idx + 1}")


def announce_player_finished(player_idx: int) -> None:
    write(f"Player{player_idx + 1} finished the game")


def print_hand(hand: list[Card | Joker]) -> None:
    write(f"{ANSI_CYAN}Your hand:{ANSI_RESET}")
    for i, card in enumerate(hand):
        write(f"  {i}: {card}")


def print_groups(hand: list[Card | Joker]) -> None:
    cards = [c for c in hand if not isinstance(c, Joker)]
    jokers = [c for c in hand if isinstance(c, Joker)]
    by_num = {}
    for c in cards:
        by_num.setdefault(c.num, 0)
        by_num[c.num] += 1

    singles = sorted([int(n) for n, cnt in by_num.items() if cnt >= 1])
    pairs = sorted([int(n) for n, cnt in by_num.items() if cnt >= 2])
    triples = sorted([int(n) for n, cnt in by_num.items() if cnt >= 3])
    quads = sorted([int(n) for n, cnt in by_num.items() if cnt >= 4])

    write(f"{ANSI_CYAN}Playable groups (no jokers):{ANSI_RESET}")
    write(f"  Singles: {singles if singles else '-'}")
    write(f"  Pairs: {pairs if pairs else '-'}")
    write(f"  Triples: {triples if triples else '-'}")
    write(f"  Quads: {quads if quads else '-'}")
    if jokers:
        write(
            f"{ANSI_YELLOW}Jokers in hand: {len(jokers)} "
            f"(you can use them to extend groups if you want){ANSI_RESET}"
        )
