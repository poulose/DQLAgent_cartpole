import gymnasium as gym
import tensorflow as tf
from tensorflow import keras# Create the environment with render_mode set to "human"
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
state, _ = env.reset()

'''
# Run a simple episode
for _ in range(10000):
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    env.render()  # This displays the animation

    if terminated or truncated:
        break

env.close()
'''


model = keras.load_model("cartpole1000.keras")


print(model.summary())

for _ in range(10000):
    action = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()
