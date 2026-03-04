from president.player import Player
from president.rl.agent import Agent, LinearAgent, MLPAgent, load_agent
from president.strategy import AgentStrategy, Smallest, UserStrategy
from president.table import Table


p1 = Player("Bot1", Smallest())

p2 = Player("Bot2", Smallest())

agent = load_agent("./MLPA_7_3.npz")
a1 = Player("MLP1", AgentStrategy(agent))

u = Player("You", UserStrategy())

t = Table([p1, p2, a1, u])

t.game()
