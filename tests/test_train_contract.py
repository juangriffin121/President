import numpy as np

from negro.rl.agent import LinearAgent
from negro.rl.train import test as eval_test
from negro.rl.train import train


def test_train_returns_logs_with_expected_lengths() -> None:
    agent = LinearAgent()
    _, log = train(num_games=6, agent=agent)

    assert len(log.rewards) == 6
    assert len(log.hand_predictions) == 6
    assert len(log.predictor_right) == 6


def test_train_anneals_dt_and_temperature_at_expected_intervals() -> None:
    agent = LinearAgent()
    initial_dt = agent.dt
    initial_temp = agent.temperature

    train(num_games=401, agent=agent)

    expected_dt = max(0.1 * initial_dt, initial_dt * 0.9 * 0.9)
    expected_temp = max(0.25 * initial_temp, initial_temp * 0.9 * 0.9)
    assert np.isclose(agent.dt, expected_dt)
    assert np.isclose(agent.temperature, expected_temp)


def test_eval_test_returns_expected_log_length_and_freezes_agent() -> None:
    agent = LinearAgent()
    log = eval_test(num_games=7, agent=agent)

    assert len(log.rewards) == 7
    assert len(log.hand_predictions) == 7
    assert len(log.predictor_right) == 7
    assert agent.frozen is True
