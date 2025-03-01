from game_2048_env import Game2048Env

env = Game2048Env()
obs = env.reset()

while True:
    print("\nGrille actuelle :")
    print(env.grid)
    
    action = input("Entrez un mouvement (0=Haut, 1=Bas, 2=Gauche, 3=Droite, q=Quitter) : ")
    if action == "q":
        break
    
    try:
        action = int(action)
        obs, reward, done, _ = env.step(action)
        print(f"Récompense : {reward}")
        if done:
            print("⚠️ GAME OVER ⚠️")
            break
    except ValueError:
        print("❌ Entrez un nombre entre 0 et 3.")
