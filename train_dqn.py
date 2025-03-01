import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from game_2048_env import Game2048Env  # Ton environnement qui renvoie une observation de forme (16,4,4)
import numpy as np

# Créer l'environnement
env = Game2048Env()

## Fonction d'évaluation du modèle
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
    print(f"Score sur {episodes} épisodes, Meilleur score : {max(scores)}, Score moyen : {avg_score}, Médiane : {np.median(scores)}")
    return avg_score

# Définir l'architecture du réseau via policy_kwargs
policy_kwargs = dict(net_arch=[2048, 1024, 512])

# Créer le modèle DQN avec CnnPolicy
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=0.00005,
    batch_size=128,
    buffer_size=50000,
    exploration_fraction=0.4,
    exploration_final_eps=0.20,
    tensorboard_log="./tensorboard_dqn/",
    device="cuda"
)

TIMESTEPS = 200000  # Ajuste selon tes ressources

log_dir = "./tensorboard_dqn/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Entraîner l'agent
model.learn(total_timesteps=TIMESTEPS)

# Évaluer le modèle sur 100 épisodes
evaluate_model(env, model, episodes=100)

# Sauvegarder le modèle
model.save("dqn_2048_cnn")

print("✅ Entraînement terminé et modèle sauvegardé.")
