from deck import Deck
from card import Card, Joker
from rules import valid_choice
from strategy import Pass
from player import Player


class Table:
    def __init__(self, players: list[Player]) -> None:
        self.deck = Deck()
        for i, player in enumerate(players):
            player.id = i
        self.players = players
        self.winners: list[Player] = []
        self.played: list[list[Card | Joker]] = []
        self.bin: list[Card | Joker] = []

    def num_players(self):
        return len(self.players)

    def deal(self):
        self.deck.shuffle()
        cards = self.deck.cards
        i = 0
        while len(cards) > 0:
            card = cards.pop()
            self.players[i].hand.append(card)
            i = (i + 1) % self.num_players()

    def round(self, starting_player_idx: int):
        i = starting_player_idx
        last_played_idx = None
        while True:
            if i == last_played_idx:
                break
            player = self.players[i]
            played = player.choose_cards([self.played, self.bin])
            if played is not None:
                last_played = self.played[-1]
                assert valid_choice(played, last_played)
                if len(player.hand) == 0:
                    winner = self.players.pop(i)
                    self.winners.append(winner)
                    # wins the game and the round ends the player next to it starts,
                    # i is now the idx of the next one, if i was the last one since there's now i players(start at 0) 10 % 10 = 0
                    return i % self.num_players()
                self.played.append(played)
                last_played_idx = i
            i = (i + 1) % self.num_players()
        winner_idx = i
        return winner_idx

    def get_starting_player(self, presidente: Player | None) -> int:
        if presidente:
            return presidente.id
        else:
            return 0

    def exchange_cards(self, presidente, negro, vise_presidente, vis_negro):
        pass

    def game(self):
        presidente = None
        negro = None
        vis_negro = None
        vise_presidente = None
        if self.winners != []:
            negro = self.winners[-1]
            presidente = self.winners[0]
            if len(self.winners) > 3:
                vis_negro = self.winners[-1]
                vise_presidente = self.winners[0]

            self.players = self.winners.copy()
            self.winners = []
            self.players.sort(key=lambda x: x.id)
        self.deal()
        self.exchange_cards(presidente, negro, vise_presidente, vis_negro)
        starting_player_idx = self.get_starting_player(presidente)
        while self.num_players() > 0:
            starting_player_idx = self.round(starting_player_idx)

    def __repr__(self) -> str:
        txt = f"Players:\n"
        for player in self.players:
            txt += player.__repr__()
        txt += f"Played:\n\t{self.played}\n"
        txt += f"Bin:\n\t{self.bin}\n"
        return txt


p1 = Player("p1", Pass())
p2 = Player("p2", Pass())
p3 = Player("p3", Pass())
t = Table([p1, p2, p3])
print(t)
t.deal()
print(t)
