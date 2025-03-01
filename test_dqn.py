import numpy as np
from stable_baselines3 import DQN
from game_2048_env import Game2048Env
from collections import defaultdict

# Charger l’environnement et le modèle entraîné
env = Game2048Env()
model = DQN.load("dqn_2048_cnn.zip")  # Assure-toi que le fichier du modèle est bien présent

# Tester sur 1000 parties
n_games = 1000
tile_counts = defaultdict(int)

for i in range(n_games):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    # Trouver la tuile maximale atteinte
    max_tile = np.max(env.grid)
    tile_counts[max_tile] += 1

    if (i + 1) % 100 == 0:
        print(f"Progression : {i+1}/{n_games} parties terminées...")

tile_order = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
results = {tile: (tile_counts[tile] / n_games) * 100 for tile in tile_order}

# Écriture des résultats dans un fichier
with open("result.txt", "w") as f:
    f.write("Max Tile | DQN_NUL (%)\n")
    f.write("----------------------\n")
    for tile in tile_order:
        f.write(f"{tile:<8} | {results[tile]:.1f} %\n")

print("✅ Test terminé ! Résultats enregistrés dans `result.txt`.")
