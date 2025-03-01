import gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from game_2048_env import Game2048Env

# Charger l’environnement et le modèle entraîné
env = Game2048Env()
model = PPO.load("ppo_2048.zip")

obs = env.reset()
done = False

total_score = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_score += reward
    print(f"Action : {action}, Récompense : {reward}, Score Total : {total_score}")

