import os
import json
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
from episode_visualizer import show_episode_outcome
from Environment.prey_predator_Env import PreyPredatorEnv
from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


"""
This file contains the visualization and evaluation of the performance of 
trained cybernetic agents. 
"""

# Smooth the rewards using a moving average
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

# === 1. Relative paths ===
PROJECT_ROOT = os.getcwd()                          
JSONS_DIR     = os.path.join(PROJECT_ROOT, "jsons")
MODEL_DIR     = os.path.join(PROJECT_ROOT, "MAPPO_Model_RL", "best_model")

# === 2. Import env ===
sys.path.append(os.path.join("src", "Visualization"))


tune.register_env(
    "prey_predator_mappo_env",
    lambda cfg: PettingZooEnv(parallel_to_aec(PreyPredatorEnv(render_mode="human")))
)

# === 3. Load the trained model ===
algo = Algorithm.from_checkpoint(MODEL_DIR)

# === 4 y 5. Evaluar múltiples episodios si se desea ===
while True:
    # === (Paso 4: episodio + render) ===
    end_cause = "unknown"
    env = PreyPredatorEnv(render_mode="human")
    obs, _ = env.reset()
    terminated = {agent: False for agent in env.possible_agents}
    episode_rewards = {agent: 0.0 for agent in env.possible_agents}
    truncated = {agent: False for agent in env.possible_agents}

    while not all(terminated.values()) and not all(truncated.values()):
        actions = {}
        for agent in env.agents:
            if not terminated[agent]:
                pol_id = algo.config.policy_mapping_fn(agent, None)
                act = algo.compute_single_action(obs[agent], policy_id=pol_id)
                actions[agent] = act
        obs, rewards, terminated, truncated, infos = env.step(actions)
        final_infos = infos

        env.render()
        for a, r in rewards.items():
            episode_rewards[a] += r

    for info in infos.values():
        if "cause" in info:
            end_cause = info["cause"]
            break
    env.close()

    print("\n🎯 Recompensas del episodio:")
    for a, r in episode_rewards.items():
        print(f"  {a}: {r:.2f}")

    # === (Paso 5: visualizar resultado del episodio) ===
    action = show_episode_outcome(episode_rewards["prey"], episode_rewards["predator"], PROJECT_ROOT, final_infos)

    if action == "next":
        continue   # otro episodio
    else:
        break      # salir a mostrar gráficos finales


# === 6. Plot history from jsons/agent_rewards.json ===
rewards_path = os.path.join(JSONS_DIR, "agent_rewards.json")
if os.path.exists(rewards_path):
    with open(rewards_path, "r") as f:
        data = json.load(f)

    prey_r      = data.get("prey", [])
    predator_r  = data.get("predator", [])
    episodes    = list(range(len(prey_r)))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, prey_r,      color="blue",    label="Prey",     linewidth=1.5)
    plt.plot(episodes, predator_r,  color="orange",  label="Predator", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode (MAPPO)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ No se encontró jsons/agent_rewards.json.")

# === 7. Plot with smoothing ===
rewards_path = os.path.join(JSONS_DIR, "agent_rewards.json")
if os.path.exists(rewards_path):
    with open(rewards_path, "r") as f:
        data = json.load(f)

    prey_r      = data.get("prey", [])
    predator_r  = data.get("predator", [])
    episodes    = list(range(len(prey_r)))

    window_size = 50
    prey_smooth     = moving_average(prey_r, window_size)
    predator_smooth = moving_average(predator_r, window_size)
    smooth_episodes = list(range(window_size - 1, len(prey_r)))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, prey_r, color="blue", alpha=0.2, label="Prey (raw)", linewidth=1)
    plt.plot(episodes, predator_r, color="orange", alpha=0.2, label="Predator (raw)", linewidth=1)
    plt.plot(smooth_episodes, prey_smooth, color="blue", label="Prey (smoothed)", linewidth=2)
    plt.plot(smooth_episodes, predator_smooth, color="orange", label="Predator (smoothed)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode (MAPPO) - Smoothed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ No se encontró jsons/agent_rewards.json.")
