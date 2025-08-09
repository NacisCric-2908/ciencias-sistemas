import os
import ray
import shutil
import datetime
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from MAPPO_agent_reward_logger import AgentRewardLogger
from MAPPO_prey_predator_Env_rllib import env_creator

"""
This file contains the training of the MAPPO algorithm for the Prey-Predator environment.
"""

# === 1. Start Ray ===
ray.init(local_mode=True, include_dashboard=False)
os.environ["RLLIB_NUM_GPUS"] = "0"

# === 2. Register environment ===
env_name = "prey_predator_mappo_env"
register_env(env_name, env_creator)

# === 3. Mapping policies ===
def policy_mapping_fn(agent_id, episode, **kwargs):
    return agent_id  # "prey" o "predator"


# === 4. Configuración PPO ===
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .training(train_batch_size=4000, gamma=0.95, lr=1e-4, clip_param=0.2)
    .multi_agent(
        policies={
            "prey": (None, env_creator().observation_space["prey"], env_creator().action_space["prey"], {}),
            "predator": (None, env_creator().observation_space["predator"], env_creator().action_space["predator"], {}),
        },
        policy_mapping_fn=policy_mapping_fn,
    )
    .callbacks(AgentRewardLogger)
)

# === 5. Train ===
print("\n🚀 Iniciando entrenamiento MAPPO...")

results = tune.run(
    "PPO",
    name="MAPPO_train",
    stop={"episodes_total": 50000},
    config=config.to_dict(),
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max"
)

# === 6. Save best checkpoint ===
best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
checkpoint_path = best_trial.checkpoint.path

model_dir = os.path.join("MAPPO_Model_RL")
os.makedirs(model_dir, exist_ok=True)
model_checkpoint_path = os.path.join(model_dir, "best_model")
shutil.copytree(checkpoint_path, model_checkpoint_path, dirs_exist_ok=True)

with open("last_mappo_checkpoint.txt", "w") as f:
    f.write(model_checkpoint_path)

print(f"\n✅ ENTRENAMIENTO COMPLETADO")
print(f"🔹 Checkpoint guardado en: {model_checkpoint_path}")

# === 7. Save result.json ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_dir = os.path.join("jsons")
os.makedirs(json_dir, exist_ok=True)

trial_dir = best_trial.logdir
result_json_src = os.path.join(trial_dir, "result.json")
result_json_dst = os.path.join(json_dir, f"mappo_result_{timestamp}.json")

if os.path.exists(result_json_src):
    shutil.copy(result_json_src, result_json_dst)
    print(f"📁 result.json copiado en: {result_json_dst}")
else:
    print("❌ No se encontró result.json")

# === 8. Copy agent_rewards.json from ray_results to project ===
rewards_src_path = os.path.join(trial_dir, "jsons", "agent_rewards.json")
rewards_dst_path = os.path.join("jsons", f"agent_rewards_{timestamp}.json")

if os.path.exists(rewards_src_path):
    shutil.copy(rewards_src_path, rewards_dst_path)
    print(f"📊 agent_rewards.json copiado en: {rewards_dst_path}")
else:
    print("❌ [ADVERTENCIA] No se encontró 'agent_rewards.json' dentro de ray_results.")

