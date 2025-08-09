# plot_smoothing.py
import os
import json
import matplotlib.pyplot as plt

# === Parámetros ===
JSONS_DIR = "jsons"
REWARDS_FILE = "agent_rewards.json"

# === Función para suavizar (media móvil) ===
def moving_average(data, window_size):
    if len(data) < window_size:
        return []
    return [
        sum(data[i-window_size+1:i+1]) / window_size
        for i in range(window_size-1, len(data))
    ]

# === Rutas ===
rewards_path = os.path.join(JSONS_DIR, REWARDS_FILE)

if not os.path.exists(rewards_path):
    print(f"❌ No se encontró {rewards_path}.")
    exit()

# === Cargar datos ===
with open(rewards_path, "r") as f:
    data = json.load(f)

prey_r     = data.get("prey", [])
predator_r = data.get("predator", [])

# === Lista de límites de episodios a graficar ===
episode_limits = [100, 1000, 5000, 10000, 20000, 30000, 40000, len(prey_r)]

# === Ventana de suavizado ===
window_size = 50

for limit in episode_limits:
    # Recortar datos
    prey_cut     = prey_r[:limit]
    predator_cut = predator_r[:limit]
    episodes     = list(range(len(prey_cut)))

    # Calcular suavizado
    prey_smooth     = moving_average(prey_cut, window_size)
    predator_smooth = moving_average(predator_cut, window_size)
    smooth_episodes = list(range(window_size - 1, len(prey_cut)))

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, prey_cut, color="blue", alpha=0.2, label="Prey (raw)", linewidth=1)
    plt.plot(episodes, predator_cut, color="orange", alpha=0.2, label="Predator (raw)", linewidth=1)
    if prey_smooth:
        plt.plot(smooth_episodes, prey_smooth, color="blue", label="Prey (smoothed)", linewidth=2)
    if predator_smooth:
        plt.plot(smooth_episodes, predator_smooth, color="orange", label="Predator (smoothed)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Total Reward per Episode (MAPPO) - Smoothed\n{limit} iteraciones")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # <- Espera que cierres para pasar al siguiente
