import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from game_2048_env import Game2048Env  # Notre environnement Gym
import numpy as np

# Créer l’environnement
env = Game2048Env()

## evaluation du modele 
def evaluate_model(env, model, episodes):
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_score += reward
        scores.append(total_score)
    avg_score = np.mean(scores)
    print(f"Score sur {episodes} parties, Meilleur score : {max(scores)}, Score moyen : {np.mean(scores)}, Médiane : {np.median(scores)}")
    return avg_score


model = DQN("MlpPolicy", 
            env, 
            policy_kwargs=dict(net_arch=[1024, 1024, 512]),
            verbose=1,
            learning_rate=0.00005,
            batch_size=64,  # ✅ Meilleure généralisation
            buffer_size=100000,  # ✅ Meilleure mémoire
            exploration_fraction=0.99,  # ✅ Réduit l'exploration trop longue
            exploration_final_eps=0.20,
            tensorboard_log="./tensorboard_logs/",
            device="cuda")



# Entraîner l’agent
TIMESTEPS = 1000000 # Ajuste selon la puissance de ton PC

log_dir = "./tensorboard_logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=TIMESTEPS)

evaluate_model(env, model, episodes=100)
# Sauvegarder le modèle
model.save("ppo_2048")

print("✅ Entraînement terminé et modèle sauvegardé.")
