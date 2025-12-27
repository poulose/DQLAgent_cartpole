# Imports for DQL agent
import os
import random
import warnings
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow import keras
from
# Imports for random agent
import human_gym_env as human_render
import gymnasium as gym_env

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

if __name__ == "__main__":
    #main()

    randomAgent = RandomAgent()
    episodes_count = input("How many episodes would you like to run? ") # Max steps are 200 to 500 steps depending on cartpole env.
    results = randomAgent.act(int(episodes_count))
    print(f"The results and rewards : {results}")
