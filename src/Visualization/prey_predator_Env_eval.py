import os
import json
import time
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv as RllibWrapper
from prey_predator_Env_rllib import env as prey_predator_env


# === 1. Register the environment for RLlib ===
register_env("prey_predator_env", lambda config: ParallelPettingZooEnv(prey_predator_env(render_mode="human")))

# === 2. Load trained model from checkpoint ===
checkpoint_path = "./PPO_Model_RL/best_model"
print(f"\n🔄 Cargando modelo entrenado desde: {checkpoint_path}")
trained_algo = Algorithm.from_checkpoint(checkpoint_path)

# === 3. Create env instance with render_mode="human" ===
env_render = RllibWrapper(prey_predator_env(render_mode="human"))
obs, _ = env_render.reset()
terminated = {agent: False for agent in obs}

print("\n🎥 Iniciando simulación...")

for _ in range(500):
    actions = {
        agent_id: trained_algo.compute_single_action(
            agent_obs,
            policy_id="prey_policy" if agent_id == "prey" else "predator_policy"
        )
        for agent_id, agent_obs in obs.items()
    }

    obs, _, terminated, _, _ = env_render.step(actions)
    env_render.render()

    if all(terminated.values()):
        print("🏁 Episodio terminado.")
        break

    time.sleep(0.05)

env_render.close()
print("✅ SIMULACIÓN COMPLETADA.")


# === 4. Mostrar gráfica de recompensas ===
rewards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../jsons/agent_rewards.json"))

if os.path.exists(rewards_path):
    with open(rewards_path, "r") as f:
        rewards_data = json.load(f)

    episodes = list(range(len(rewards_data["prey"])))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_data["prey"], label="Prey", color="blue")
    plt.plot(episodes, rewards_data["predator"], label="Predator", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("❌ No se encontró el archivo agent_rewards.json.")
