import gymnasium as gym
import numpy as np
from typing import Tuple
import imageio

env = gym.make("Blackjack-v1", sab=False, render_mode="rgb_array")

space_size = [env.observation_space[i].n for i in range(len(env.observation_space))]

q_array = np.zeros((*space_size, env.action_space.n))

epsilon = 1.0
epsilon_min = 0.0001
epsilon_decay = 0.999999
learning_rate = 0.001


def greedy_epsilon(state: Tuple[int]) -> int:
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_array[state])

    return action


def greedy_action(state: Tuple[int]) -> int:
    return np.argmax(q_array[state])


def evaluate_policy(n_episodes: int = 500) -> float:
    scores = []
    max_steps = 100

    for e in range(n_episodes):
        state, _ = env.reset()
        rewards = 0
        for s in range(max_steps):
            action = greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            if terminated or truncated:
                scores.append(rewards)
                break

    return np.mean(scores), np.std(scores)


def train(n_episodes: int = 100) -> None:
    global epsilon

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

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if terminated or truncated:
                break

        if (e % 50000) == 0:
            mn_score, std_score = evaluate_policy()
            print("Mean score {:.2f}".format(mn_score))
            print("Std score {:.2f}".format(std_score))
            print("Temperature: {:.3f}".format(epsilon))

    return None


def record_video(env, policy, out_directory, fps=1, n_games=30):
    """Generate a replay video of the agent :param env :param Qtable:
    Qtable of our agent :param out_directory :param fps: how many
    frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []

    for g in range(n_games):
        state = env.reset()[0]
        img = env.render()
        images.append(img)
        terminated = False

        while not terminated:
            # Take the action (index) that have the maximum expected
            # future reward given that state
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(
                action
            )  # We directly put next_state = state for recording logic
            img = env.render()
            images.append(img)
        
    imageio.mimsave(
        out_directory, [np.array(img) for img in images], fps=fps
    )


if __name__ == "__main__":
    train(10**6)
    record_video(env, greedy_action, "./data/blackjack_v0.mp4")
    
