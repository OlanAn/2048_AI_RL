import numpy as np
import gym
from gym import spaces

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.grid_size = 4
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size ** 2,), dtype=np.float32
        )

        # Pour gérer les mouvements invalides répétés
        self.invalid_action_streak = 0
        self.last_invalid_action = None

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.add_random_tile()
        self.add_random_tile()

        # Réinitialise la gestion des invalides
        self.invalid_action_streak = 0
        self.last_invalid_action = None

        return self.get_observation()

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.choice(len(empty_cells))]
            self.grid[x, y] = 2 if np.random.rand() < 0.9 else 4

    def _simulate_move(self, grid, action):
        """Simule un mouvement et retourne (new_grid, reward)."""
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
        """Fusionne les tuiles (lignes) de gauche à droite et calcule la récompense."""
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

            new_grid[i, : len(new_line)] = new_line

        return new_grid, total_reward

    def no_more_moves(self):
        """Retourne True si aucun mouvement n’est possible dans aucune direction."""
        for action in [0, 1, 2, 3]:
            new_grid, _ = self._simulate_move(self.grid, action)
            if not np.array_equal(new_grid, self.grid):
                return False
        return True

    def step(self, action):
        old_grid = self.grid.copy()
        new_grid, move_reward = self._simulate_move(old_grid, action)

        if np.array_equal(old_grid, new_grid):
            # ➜ Mouvement invalide
            self.grid = old_grid

            # Incrémente la streak si c’est la même action qu’avant
            if self.last_invalid_action == action:
                self.invalid_action_streak += 1
            else:
                self.invalid_action_streak = 1

            # Calcul de la pénalité en fonction du nombre de répétitions
            #  1ère fois  : -1
            #  2ème fois  : -5
            #  3ème fois  : -10
            #  4ème fois  : -15, etc.
            if self.invalid_action_streak == 1:
                reward = -1
            else:
                reward = -5 * self.invalid_action_streak

            self.last_invalid_action = action

            # Vérifie si plus aucun coup n’est possible
            if self.no_more_moves():
                done = True
                # Pénalité supplémentaire si pas de tuile 2048
                if np.max(self.grid) < 2048:
                    reward -= 50
            else:
                done = False

        else:
            # ➜ Mouvement valide
            self.grid = new_grid
            reward = move_reward

            # Réinitialise la streak d’actions invalides
            self.invalid_action_streak = 0
            self.last_invalid_action = None

            # Ajout d’une tuile
            self.add_random_tile()

            # Vérifie si la grille est bloquée
            if self.no_more_moves():
                done = True
                if np.max(self.grid) < 2048:
                    reward -= 50
            else:
                done = False

        return self.get_observation(), float(reward), done, {}

    def get_observation(self):
        """Vecteur normalisé [0..1]."""
        return (self.grid.flatten() / 2048.0).astype(np.float32)
