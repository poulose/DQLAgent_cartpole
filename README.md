# Reinforcement learning Finance - cartpole
    Cartpole environment gives the opportunity to interact with a game environment by updating parameters
## Requirement - pygame support missing Python 3.14 (Latest)
    16.12.2025 - The Cartpole environment works best with Python 3.13, especially if running in human render mode. pin version 3.13 and toml for update of environment
    Cartpole human render example : human_gym_env.py
## Random Agent
    A non rendered env., a random agent which receives reward by chance
## DQL Agent
    A non rendered env., a DQL agent which receives reward and explores based on hyper parameters provided by training over multiple episodes. The models trained are from 500 to 2000 episodes. Optimal at 1000 eps.
## DQL Agent - Arm system requirement - branch arm-DQLAgent
    Requires py 3.12
## Visual test & Review
    A human rendered env. to find optimal episodes and iteration for human review. human_gym_env.py uses trained models to review the results. Identified issues in original test implementation. To be reviewed and fixed. 