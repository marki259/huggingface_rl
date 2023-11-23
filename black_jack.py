import gymnasium as gym
import numpy as np
from typing import Tuple

env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")

space_size = [env.observation_space[i].n for i in range(len(env.observation_space))]

q_array = np.zeros((*space_size, env.action_space.n))

epsilon = 1
epsilon_min = 0.001
epsilon_decay = 0.99
learning_rate = 0.001


def greedy_epsilon(state: Tuple[int]) -> int:
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_array[state])

    return action


def greedy_action(state: Tuple[int]) -> int:
    return np.argmax(q_array[state])

def evaluate_policy(n_episodes: int = 10) -> float:
    
    

def train(n_episodes: int = 100) -> None:
    max_steps = 100

    for e in range(n_episodes):
        state, _ = env.reset()

        for s in range(max_steps):
            action = greedy_epsilon(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # update
            q_array[*state, action] += learning_rate * (
                reward + np.max(q_array[new_state]) - q_array[*state, action]
            )

            if (terminated or truncated):
                break

    return None


if __name__ == "__main__":
    print("Hello")
