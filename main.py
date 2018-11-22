"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from trainer import Trainer
from policy import HillClimbingPolicy
from replay_memory import ReplayMemory
from hill_climbing_env import HillClimbingEnv
from mcts import execute_episode


def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == '__main__':
    n_actions = 4
    n_obs = 49

    trainer = Trainer(lambda: HillClimbingPolicy(n_obs, 20, n_actions))
    network = trainer.step_model

    mem = ReplayMemory(200,
                       { "ob": np.long,
                         "pi": np.float32,
                         "return": np.float32},
                       { "ob": [],
                         "pi": [n_actions],
                         "return": []})

    def test_agent(iteration):
        test_env = HillClimbingEnv()
        total_rew = 0
        state, reward, done, _ = test_env.reset()
        step_idx = 0
        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, _ = network.step(np.array([state]))
            # print(p)
            action = np.argmax(p)
            state, reward, done, _ = test_env.step(action)
            step_idx+=1
            total_rew += reward
        log(test_env, iteration, step_idx, total_rew)

    value_losses = []
    policy_losses = []

    for i in range(1000):
        if i % 50 == 0:
            test_agent(i)
            plt.plot(value_losses, label="value loss")
            plt.plot(policy_losses, label="policy loss")
            plt.legend()
            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 32,
                                                                 HillClimbingEnv)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()

        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)
