import os
import random
import warnings
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import gymnasium as gym_env

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class DQNAgent:
    """
    DQN with:
      - target network (stability)
      - batched replay (speed)
      - training every step after warmup
      - step-based epsilon decay (reaches exploitation faster)
      - Huber loss (often more stable than MSE)
    """

    def __init__(
            self,
            env_name="CartPole-v1",
            gamma=0.99,
            lr=5e-4,
            batch_size=64,
            memory_size=50_000,
            warmup_steps=1_000,
            train_every=1,
            target_update_every=1_000,
            eps_start=1.0,
            eps_min=0.05,
            eps_decay=0.9995,
    ):
        self.env = gym_env.make(env_name)
        self.state_dim = int(self.env.observation_space.shape[0])  # 4
        self.n_actions = int(self.env.action_space.n)

        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.warmup_steps = warmup_steps
        self.train_every = train_every
        self.target_update_every = target_update_every

        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.total_steps = 0
        self.episode_rewards = []
        self.max_reward = 0

        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target()

    def _build_model(self, lr: float):
        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(self.state_dim,)),
                Dense(128, activation="relu"),
                Dense(self.n_actions, activation="linear"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.Huber(),
        )
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> int:
        # state shape: (1, state_dim)
        if random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        q = self.model(state, training=False).numpy()
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.vstack([b[0] for b in batch]).astype(np.float32)       # (B,4)
        actions = np.array([b[1] for b in batch], dtype=np.int32)          # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)        # (B,)
        next_states = np.vstack([b[3] for b in batch]).astype(np.float32)  # (B,4)
        dones = np.array([b[4] for b in batch], dtype=np.float32)          # (B,) 0/1

        # Q(s, ·)
        q_values = self.model(states, training=False).numpy()              # (B,2)

        # max_a' Q_target(s', a')
        q_next = self.target_model(next_states, training=False).numpy()    # (B,2)
        max_q_next = np.max(q_next, axis=1)                                # (B,)

        # Bellman target for chosen actions
        targets = q_values.copy()
        targets[np.arange(self.batch_size), actions] = rewards + (1.0 - dones) * self.gamma * max_q_next

        # One batched update
        self.model.fit(states, targets, epochs=1, verbose=0)

    def learn(self, episodes: int, max_steps_per_episode: int = 500):
        for ep in range(1, episodes + 1):
            obs, _ = self.env.reset(seed=SEED + ep)
            state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

            ep_reward = 0.0

            for t in range(1, max_steps_per_episode + 1):
                self.total_steps += 1

                action = self.act(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)

                next_state = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)

                self.remember(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

                # Train every step after warmup (or every N steps)
                if self.total_steps >= self.warmup_steps and (self.total_steps % self.train_every == 0):
                    self.replay()

                # Update target net periodically
                if self.total_steps % self.target_update_every == 0:
                    self.update_target()

                # Decay epsilon per step (fast enough to exploit within a few thousand steps)
                if self.epsilon > self.eps_min:
                    self.epsilon *= self.eps_decay
                    if self.epsilon < self.eps_min:
                        self.epsilon = self.eps_min

                if done:
                    break

            self.episode_rewards.append(ep_reward)
            self.max_reward = max(self.max_reward, int(ep_reward))

            # Simple rolling average for signal
            window = 50
            recent = self.episode_rewards[-window:]
            avg_recent = sum(recent) / len(recent)

            print(
                f"episode={ep:4d} | reward={int(ep_reward):3d} | "
                f"avg{window}={avg_recent:6.1f} | max={self.max_reward:3d} | eps={self.epsilon:0.3f}"
            )

    def test(self, episodes: int = 10, max_steps_per_episode: int = 500):
        saved_eps = self.epsilon
        self.epsilon = 0.0  # pure exploitation

        scores = []
        for ep in range(1, episodes + 1):
            obs, _ = self.env.reset(seed=999 + ep)
            state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

            total = 0
            for _ in range(max_steps_per_episode):
                q = self.model(state, training=False).numpy()
                action = int(np.argmax(q[0]))
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                state = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
                total += reward
                if terminated or truncated:
                    break
            scores.append(int(total))

        self.epsilon = saved_eps
        print("test scores:", scores)
        print(f"test avg: {sum(scores)/len(scores):.1f}")


if __name__ == "__main__":
    episodes_count = int(input("How many episodes would you like to run? "))

    agent = DQNAgent(
        lr=5e-4,
        batch_size=64,
        warmup_steps=1_000,
        train_every=1,
        target_update_every=1_000,
        eps_decay=0.9995,
        eps_min=0.05,
    )

    start = time.perf_counter()
    agent.learn(episodes_count)
    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.3f}s")

    agent.test(episodes=15)

    print(f"{agent.epsilon:.6f} epsilon")

    agent.model.save(f"cartpole_dqn{episodes_count}.keras")
    print(f"Saved model to cartpole_dqn{episodes_count}.keras")
