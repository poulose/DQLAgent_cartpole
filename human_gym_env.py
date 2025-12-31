import time
import numpy as np
import gymnasium as gym
from tensorflow import keras

MODEL_PATH = "cartpole_dqn1000_optimal.keras"  # must match what you saved in main.py and number of episodes used
# available with 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000

# Create the environment with render_mode set to "human"
env = gym.make("CartPole-v1", render_mode="human")

'''# Reset the environment
state, _ = env.reset()

# Run a simple episode
for _ in range(10000):
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    env.render()  # This displays the animation

    if terminated or truncated:
        break

'''
# Run simulation using a trained model
# Load trained model
model = keras.models.load_model(MODEL_PATH)

# Reset environment
obs, _ = env.reset()
state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

# Run a policy episode
total_reward = 0
for step in range(1000):
    # Greedy action from Q-network
    q = model(state, training=False).numpy()
    action = int(np.argmax(q[0]))

    obs, reward, terminated, truncated, info = env.step(action)
    state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    #print(f"The  step: {step} finished with reward {reward}")

    total_reward += reward

    # Small sleep to make it watchable
    time.sleep(0.02)

    if terminated or truncated:
        obs, _ = env.reset()
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        print(f"The episode finished after {step} steps, with total reward {total_reward}")
        total_reward = 0
        time.sleep(0.2)


env.close()
