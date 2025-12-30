import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py  # For inspecting .keras files
import time

episodes = int(input("Provide model episode number between range 500 and 4000: "))

if episodes:
    episodes = round((episodes/500) * 500)



model_path = f"cartpole_dqn{episodes}.keras" # use 1000 and above with steps of 500. Higher trained ones do not have better performance when observed with human render

# First, inspect the model architecture to get exact layer sizes
print("Inspecting model architecture...")

# Method 1: Try to load model config only (no weights)
try:
    with h5py.File(model_path, 'r') as f:
        config = f['model_config'].value.decode('utf-8')
        print("Model config:", config)
except Exception as e:
    print(f"Could not read model config: {e}")

# Method 2: Recreate based on error info - FIRST LAYER IS 128 UNITS
print("\nRecreating model with correct architecture (first layer: 128 units)...")

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(4,)),  # Fixed: 128 units
    keras.layers.Dense(128, activation='relu'),  # Typical DQN has 2 hidden layers
    keras.layers.Dense(2, activation='linear')  # 2 actions for CartPole
])

print(model.summary())

# Skip loading weights since architecture doesn't match exactly
print("\nSkipping weights - using random initialized model for demo")
print("The agent may not perform well without trained weights.")

# Create environment
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
state = np.asarray(obs, dtype=np.float32).reshape(1, -1)

total_reward = 0
print("\nStarting human rendering (random agent - press Ctrl+C to stop)...")

try:
    for step in range(1000):  # Reduced steps since random agent
        q_values = model(state, training=False).numpy()
        action = int(np.argmax(q_values[0]))

        obs, reward, terminated, truncated, info = env.step(action)
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        total_reward += reward

        time.sleep(0.02)

        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps, reward: {total_reward}")
            obs, info = env.reset()
            state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            total_reward = 0

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    env.close()

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