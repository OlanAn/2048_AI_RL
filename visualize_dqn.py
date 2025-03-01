import pygame
from stable_baselines3 import PPO
from game_2048_env import Game2048Env

# Initialiser Pygame
pygame.init()

# Paramètres de l'affichage
SIZE = 4
TILE_SIZE = 100
GAP_SIZE = 10
MARGIN = 20
SCREEN_SIZE = SIZE * TILE_SIZE + (SIZE + 1) * GAP_SIZE + 2 * MARGIN
SCREEN_WIDTH = SCREEN_SIZE
SCREEN_HEIGHT = SCREEN_SIZE
BACKGROUND_COLOR = (255, 251, 240)
FONT = pygame.font.SysFont('arial', 40)

# Charger le modèle et l'environnement
env = Game2048Env()
model = PPO.load("ppo_2048")

# Initialiser Pygame
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2048 - IA DQN")
clock = pygame.time.Clock()

# Fonction pour dessiner le jeu
def draw_board(screen, board):
    screen.fill(BACKGROUND_COLOR)
    for row in range(SIZE):
        for col in range(SIZE):
            value = board[row][col]
            color = (205, 192, 180) if value == 0 else (255 - value % 255, 255 - (value % 255) // 2, 180)
            rect = pygame.Rect(MARGIN + GAP_SIZE + col * (TILE_SIZE + GAP_SIZE),
                               MARGIN + GAP_SIZE + row * (TILE_SIZE + GAP_SIZE),
                               TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)
            if value != 0:
                text = FONT.render(str(value), True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

# Lancement du jeu
obs = env.reset()
running = True
done = False

while running and not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(obs)  # L’IA choisit une action
    obs, reward, done, _ = env.step(action)

    draw_board(screen, env.grid)
    pygame.display.flip()
    clock.tick(1)  # Réduit la vitesse pour voir l'IA jouer lentement

pygame.quit()
