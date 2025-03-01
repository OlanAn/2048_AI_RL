import numpy as np
import math
import random
import gym
from gym import spaces

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.grid_size = 4
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Observation: 16 canaux, chaque canal correspond à une puissance (0 à 15)
        # On représente la grille en one-hot : la dimension des canaux est 16,
        # pour indiquer soit l'absence de tuile (canal 0 activé), soit la puissance de 2.
        self.observation_space = spaces.Box(low=0, high=1, shape=(16, self.grid_size, self.grid_size), dtype=np.float32)
        # Action space: 4 directions (0=up, 1=down, 2=left, 3=right)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return self.get_observation()

    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 4 if random.random() >= 0.9 else 2

    def get_observation(self):
        """
        Retourne un tenseur one-hot de forme (16, 4, 4).
        Pour chaque case :
          - Si la case est vide (valeur 0), le canal 0 est activé (valeur 1).
          - Sinon, on calcule la puissance de 2 (int(log2(val))) et on active le canal correspondant (limité à 15).
        """
        one_hot = np.zeros((16, self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = self.grid[i, j]
                if val == 0:
                    one_hot[0, i, j] = 1.0
                else:
                    power = int(math.log(val, 2))
                    if power > 15:
                        power = 15
                    one_hot[power, i, j] = 1.0
        return one_hot

    # Vous pouvez conserver vos autres méthodes (step, merge, up, down, left, right, no_more_moves, etc.)
    # qui définissent la logique du jeu.
    def _simulate_move(self, grid, action):
        new_grid = grid.copy()
        reward = 0
        if action == 0:  # Haut
            new_grid = new_grid.T
            new_grid, reward = self.merge(new_grid)
            new_grid = new_grid.T
        elif action == 1:  # Bas
            new_grid = new_grid.T
            new_grid = np.fliplr(new_grid)
            new_grid, tmp_reward = self.merge(new_grid)
            new_grid = np.fliplr(new_grid)
            new_grid = new_grid.T
            reward += tmp_reward
        elif action == 2:  # Gauche
            new_grid, reward = self.merge(new_grid)
        elif action == 3:  # Droite
            new_grid = np.fliplr(new_grid)
            new_grid, tmp_reward = self.merge(new_grid)
            new_grid = np.fliplr(new_grid)
            reward += tmp_reward
        return new_grid, reward

    def merge(self, grid):
        new_grid = np.zeros_like(grid)
        total_reward = 0
        for i in range(grid.shape[0]):
            line = grid[i][grid[i] != 0]
            new_line = []
            skip = False
            for j in range(len(line)):
                if skip:
                    skip = False
                    continue
                if j < len(line) - 1 and line[j] == line[j + 1]:
                    fused_value = line[j] * 2
                    new_line.append(fused_value)
                    total_reward += fused_value
                    skip = True
                else:
                    new_line.append(line[j])
            new_grid[i, :len(new_line)] = new_line
        return new_grid, total_reward

    def no_more_moves(self):
        for action in [0, 1, 2, 3]:
            new_grid, _ = self._simulate_move(self.grid, action)
            if not np.array_equal(new_grid, self.grid):
                return False
        return True

    def step(self, action):
        old_grid = self.grid.copy()
        new_grid, move_reward = self._simulate_move(old_grid, action)
        if np.array_equal(old_grid, new_grid):
            self.grid = old_grid
            # Gestion des mouvements invalides (peut être personnalisée)
            reward = +0
            done = self.no_more_moves()
        else:
            self.grid = new_grid
            reward = move_reward
            self.add_random_tile()
            done = self.no_more_moves()
        return self.get_observation(), float(reward), done, {}
