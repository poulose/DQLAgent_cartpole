# Imports for DQL agent
import os
import random
import timeit
import warnings
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow.python.framework.ops import disable_eager_execution

# Imports for random agent
#import human_gym_env as human_render
import gymnasium as gym_env


# configs for DQL agent

disable_eager_execution()

opt = keras. optimizers.legacy.Adam(learning_rate=0.0001)

random.seed(100)
tf.random.set_seed(100)


def main():
    print("Hello from cartpole!")
    try:
        #human_render # Call the imported module directly
        pass

    except Exception as e:
        print(e)

class RandomAgent:
    def __init__(self):
        self.env = gym_env.make("CartPole-v1")
        self.trewards = list()
        self.term_break = False

    def act(self, episodes):

        for _ in range(episodes):
            self.env.reset()
            for step in range(1, 10):
                a = self.env.action_space.sample()
                state, reward, done, trunc, info = self.env.step(a)

                if done:
                    self.trewards.append(step)
                    self.term_break = True
                    break  # No further calls to step, instead reset and exit loop

        return self.trewards


class DQLAgent():
    def __init__(self):
        self._create_model()
        self.env = gym_env.make("CartPole-v1")
        self.trewards = list()
        self.max_reward = 0
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=4, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=opt)
        #self.model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state)[0])

                target = self.model.predict(next_state)
                target[0, action] = reward
                self.model.fit(state, target, epochs=2, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            state, _= self.env.reset()
            state = np.reshape(state, [1, 4])
            for f in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done or trunc:
                    self.trewards.append(f)
                    self.max_reward = max(self.max_reward, f)
                    templ = f"episode={e:4d} | treward={f:4d}"
                    templ += f" | max={self.max_reward:4d}"
                    print(templ, end="\r")
                    break
            if len(self.memory) > self.batch_size:
                self.replay()
            print()

    def test(self, episodes):
        for e in range(1, episodes + 1):
            state, _= self.env.reset()
            state = np.reshape(state, [1, 4])
            for f in range(1, 5001):
                action = np.argmax(self.model.predict(state)[0])
                state, reward, done, trunc, _ = self.env.step(action)
                state = np.reshape(state, [1, 4])
                if done or trunc:
                    print(f, end=" ")
                    break

if __name__ == "__main__":
    #main()

    randomAgent = RandomAgent()
    episodes_count = input("How many episodes would you like to run? ") # Max steps are 200 to 500 steps depending on cartpole env.
    results = randomAgent.act(int(episodes_count))
    print(f"The results and rewards : {results}")

    Agent = DQLAgent()

    #Agent.learn(int(episodes_count))
    #timed = timeit.timeit("Agent.learn(int(episodes_count))",globals=globals())

    timed = timeit.timeit("Agent.learn(int(episodes_count))", globals={'Agent': Agent, 'episodes_count': episodes_count},number=1)
    #print(f"Time taken for DQLAgent: {timed}")

    print(f"{Agent.epsilon} epsilon")

    print(f"{Agent.test(15)}")

    Agent.model.save(f"cartpole_dqn{episodes_count}.keras")
