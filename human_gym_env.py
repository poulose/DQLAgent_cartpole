import gymnasium as gym
import tensorflow as tf
from tensorflow import keras # Create the environment with render_mode set to "human"
import numpy as np


# Load modal

model_path = "cartpole_dqn_1000.keras"

model = keras.models.load_model(model_path)
print(model.summary())


env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
obs, _ = env.reset()

state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

for _ in range(10000):
    q = model(state, training=False).numpy()
    action = int(np.argmax(q[0]))

    obs, reward, terminated, truncated, info = env.step(action)
    state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

    if terminated or truncated:
        obs, _ = env.reset()
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)



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



'''
model = keras.models.load_model(MODEL_PATH)

# Reset environment
obs, _ = env.reset()
state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

# Run a policy episode
for _ in range(10000):
    # Greedy action from Q-network
    q = model(state, training=False).numpy()
    action = int(np.argmax(q[0]))

    obs, reward, terminated, truncated, info = env.step(action)
    state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

    # Small sleep to make it watchable
    time.sleep(0.02)

    if terminated or truncated:
        obs, _ = env.reset()
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

'''