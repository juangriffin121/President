from president.experiments.positional_dynamics.logger import GameLogEntry, GameLogWriter
from president.experiments.positional_dynamics.table_anim import animate_trial, save_animation
from president.rl.agent import Agent
from president.rl.features import hand_feat_names
from president.rl.hand_strength import HandStrengthPredictor
from president.strategy import AgentStrategy, Smallest
from president.table import Table
from president.player import Player, set_sleep_enabled
from president.ui import writes

PLAYERS = 5
GAMES = 10000
trial_id = f"{PLAYERS}_Linear"
PATH = f"president/experiments/positional_dynamics/run{trial_id}.csv"
TRAINING_GAMES = 5000


def log(trial_id: int,game: int, table: Table) -> GameLogEntry:

    president=table.president
    vice_president=table.vice_president
    vice_scum=table.vice_scum
    scum=table.scum

    return GameLogEntry(
        trial_id,
        game_index=game,
        num_players=PLAYERS,
        president_id=president.id if president else None,
        vice_president_id=vice_president.id if vice_president else None,
        vice_scum_id=vice_scum.id if vice_scum else None,
        scum_id=scum.id if scum else None,
        winner_ids= [p.id for p in table.winners],
        hand_strength_prediction = [p.strategy.last_hand_strength_prediction for p in table.winners]
        )


writes.set_silent(True)
set_sleep_enabled(False)
a = Agent.load("Linears/agents/L__2k__S__None.npz")
players = [Player(f"Player {i}",AgentStrategy(a.clone(perturb_std=0), HandStrengthPredictor())) for i in range(PLAYERS)]
for player in players:
    player.strategy.agent.freeze()
t = Table(players)
print(t.players)
writer = GameLogWriter(PATH)

for game in range(TRAINING_GAMES):
    t.game()
    for player in t.winners:
        print(player.id, player.strategy.last_reward, player.strategy.last_hand_strength_prediction)

players = t.winners
for player in players:
    player.strategy.hand_strength_predictor.freeze()
# new table to restart roles
t = Table(players)
for game in range(GAMES):
    t.game()
    entry = log(trial_id, game, t)
    writer.write(entry)
    print(game)

for player in t.winners:
    print(player.name)
    for name, val in zip(hand_feat_names(), player.strategy.hand_strength_predictor.w):
        print(f"\t{val:.2f}\t{name}")


