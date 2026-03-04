from negro.deck import Deck
from negro.card import Card, Joker
from negro.rules import valid_choice
from negro.ranking import sort_hand
from negro.player import Player
from negro.state import GlobalState
from negro.ui import writes


class Table:
    def __init__(self, players: list[Player]) -> None:
        self.deck = Deck()
        for i, player in enumerate(players):
            player.id = i
        self.players = players
        self.total_players = len(players)
        self.winners: list[Player] = []
        self.played: list[list[Card | Joker]] = []
        self.bin: list[Card | Joker] = []
        self.presidente: Player | None = None
        self.negro: Player | None = None
        self.vice_presidente: Player | None = None
        self.vice_negro: Player | None = None

    def num_players(self):
        return len(self.players)

    def deal(self):
        self.deck.shuffle()
        cards = self.deck.cards
        i = 0
        while len(cards) > 0:
            card = cards.pop()
            self.players[i].hand.append(card)
            self.players[i].on_deal(self.total_players)
            i = (i + 1) % self.num_players()

    def round(self, starting_player_idx: int):
        i = starting_player_idx
        last_played_idx = (
            i - 1
        ) % self.num_players()  # could be None but if everyone passes it can be infinite loop, this way if even first player
        # passes and everyone passes after, the player before the first wins the round and gets to play first, not that any of this would happen
        while True:
            if i == last_played_idx:
                self.bin.extend(sum(self.played, []))
                self.played = []
                writes.announce_round_winner(i)
                break

            player = self.players[i]
            writes.announce_turn(player.name)

            global_state = GlobalState(
                played=self.played,
                bin=self.bin,
                players=[p.id for p in self.players],
                hand_sizes=[len(p.hand) for p in self.players],
                winners=[p.id for p in self.winners],
                total_players=self.total_players,
                presidente=self.presidente.id if self.presidente else None,
                vice_presidente=(
                    self.vice_presidente.id if self.vice_presidente else None
                ),
                vice_negro=self.vice_negro.id if self.vice_negro else None,
                negro=self.negro.id if self.negro else None,
            )

            played = player.choose_cards(global_state)
            writes.announce_play(played)

            try:
                last_played = self.played[-1]
            except:
                last_played = None
            if played is not None:
                assert valid_choice(played, last_played)
                if len(player.hand) == 0:
                    winner = self.players.pop(i)
                    writes.announce_player_finished(i)
                    self.winners.append(winner)
                    # wins the game and the round ends the player next to it starts,
                    # i is now the idx of the next one, if i was the last one since there's now i players(start at 0) 10 % 10 = 0
                    if self.num_players() == 0:
                        writes.write(
                            "Player was playing alone, this shouldnt be possible"
                        )
                        return 0

                    self.bin.extend(sum(self.played, []))
                    self.played = []
                    return i % self.num_players()

                self.played.append(played)
                last_played_idx = i
            i = (i + 1) % self.num_players()
        winner_idx = i
        return winner_idx

    def get_starting_player(self, presidente: Player | None) -> int:
        if presidente:
            return presidente.id
        for i, player in enumerate(self.players):
            for card in player.hand:
                if isinstance(card, Card) and card.num == 2 and card.suit == "🪙":
                    return i
        return 0

    def exchange_cards(self, presidente, negro, vise_presidente, vis_negro):
        def take_best(player, count):
            if player is None or count <= 0:
                return []
            non_jokers = [c for c in player.hand if not isinstance(c, Joker)]
            sorted_hand = sort_hand(non_jokers, joker_value=0, reverse=True)
            taken = sorted_hand[:count]
            for c in taken:
                player.hand.remove(c)
            return taken

        def take_worst(player, count):
            if player is None or count <= 0:
                return []
            choice = player.choose_worst(count)
            assert choice is not None
            assert len(choice) == count
            for c in choice:
                player.hand.remove(c)
            return choice

        # Presidente <-> Negro (2 cards)
        if presidente and negro:
            from_negro = take_best(negro, 2)
            from_presidente = take_worst(presidente, 2)
            negro.hand.extend(from_presidente)
            presidente.hand.extend(from_negro)

        # Vise-presidente <-> Vis-negro (1 card)
        if vise_presidente and vis_negro:
            from_vis_negro = take_best(vis_negro, 1)
            from_vise_presidente = take_worst(vise_presidente, 1)
            vis_negro.hand.extend(from_vise_presidente)
            vise_presidente.hand.extend(from_vis_negro)

    def game(self):
        self.presidente = None
        self.negro = None
        self.vice_negro = None
        self.vice_presidente = None
        if self.winners != []:
            self.negro = self.winners[-1]
            self.presidente = self.winners[0]
            if len(self.winners) > 3:
                self.vice_negro = self.winners[-2]
                self.vice_presidente = self.winners[1]

            self.players = self.winners.copy()
            self.winners = []
            self.players.sort(key=lambda x: x.id)
        self.deal()
        self.exchange_cards(
            self.presidente, self.negro, self.vice_presidente, self.vice_negro
        )

        starting_player_idx = self.get_starting_player(self.presidente)
        while self.num_players() > 1:  # until the last player
            starting_player_idx = self.round(starting_player_idx)
            writes.write()

        assert len(self.players) == 1
        loser = self.players.pop(0)
        self.bin.extend(loser.hand.copy())  # cards remain in loser's hand
        loser.hand = []
        self.deck.cards = self.bin.copy()
        self.bin = []
        self.winners.append(loser)

        vices = True
        n = len(self.winners)
        presidente_idx = 0
        vice_presidente_idx = 1
        vice_negro_idx = n - 2
        negro_idx = n - 1

        if vice_presidente_idx == vice_negro_idx:
            vices = False

        for i, player in enumerate(self.winners):
            is_presidente = i == presidente_idx
            is_negro = i == negro_idx
            is_vice_presidente = i == vice_presidente_idx if vices else False
            is_vice_negro = i == vice_negro_idx if vices else False

            performance = 0
            performance = 2 if is_presidente else performance
            performance = -2 if is_negro else performance
            performance = 1 if is_vice_presidente else performance
            performance = -1 if is_vice_negro else performance

            player.inform_of_results(performance)

    def __repr__(self) -> str:
        txt = f"Players:\n"
        for player in self.players:
            txt += player.__repr__()

        txt += f"Winners:\n"
        for winner in self.winners:
            txt += winner.__repr__()
        txt += f"Played:\n\t{self.played}\n"
        txt += f"Bin:\n\t{self.bin}\n"
        return txt
