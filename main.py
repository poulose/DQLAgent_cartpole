from math import trunc

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

    def act(self, episodes):

        for _ in range(episodes):
            self.env.reset()
            for step in range(1, 10):
                a = self.env.action_space.sample()
                state, reward, done, trunc, info = self.env.step(a)

                if done:
                    self.trewards.append(step)

        return self.trewards

if __name__ == "__main__":
    #main()

    randomAgent = RandomAgent()
    results = randomAgent.act(4)
    print(f"The results and rewards : {results}")


