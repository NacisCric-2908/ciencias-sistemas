from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from prey_predator_Env_rllib import env as prey_predator_env
from agent_reward_logger import AgentRewardLogger
import os
import ray
import datetime
import shutil

# === 1. Start Ray ===
ray.init(local_mode=True, include_dashboard=False)
os.environ["RLLIB_NUM_GPUS"] = "0"

# === 2. Register the environment ===
parallel_env = prey_predator_env(render_mode=None)
obs_space_prey = parallel_env.observation_space("prey")
act_space_prey = parallel_env.action_space("prey")

obs_space_predator = parallel_env.observation_space("predator")
act_space_predator = parallel_env.action_space("predator")

register_env("prey_predator_env", lambda config: ParallelPettingZooEnv(prey_predator_env(render_mode=None)))

# === 3. Define separate policies ===
policies = {
    "prey_policy": (
        None,
        obs_space_prey,
        act_space_prey,
        {}
    ),
    "predator_policy": (
        None,
        obs_space_predator,
        act_space_predator,
        {}
    )
}

# === 4. Map agents to their corresponding policies ===
def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "prey":
        return "prey_policy"
    else:
        return "predator_policy"

# === 5. PPO Config ===
config = {
    "env": "prey_predator_env",
    "framework": "torch",
    "callbacks": AgentRewardLogger,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "env_config": {},
    "num_workers": 0,
    "rollout_fragment_length": 200,
    "log_level": "ERROR",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "tf_session_args": {
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
    }
}

# === 6. Training ===
print("\n🚀 Iniciando entrenamiento...")
results = tune.run(
    "PPO",
    stop={"episodes_total": 2000},
    config=config,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max"
)

# === 7. Save best checkpoint ===
best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
checkpoint_path = best_trial.checkpoint.path

model_dir = os.path.join("PPO_Model_RL")
os.makedirs(model_dir, exist_ok=True)
model_checkpoint_path = os.path.join(model_dir, "best_model")
shutil.copytree(checkpoint_path, model_checkpoint_path, dirs_exist_ok=True)

with open("last_checkpoint.txt", "w") as f:
    f.write(model_checkpoint_path)

print("\n✅ TRAINING COMPLETED")
print(f"🔹 Checkpoint saved in: {model_checkpoint_path}")

# === 8. Copy result.json ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_dir = os.path.join("jsons")
os.makedirs(json_dir, exist_ok=True)

trial_dir = best_trial.logdir
result_json_src = os.path.join(trial_dir, "result.json")
result_json_dst = os.path.join(json_dir, f"result_{timestamp}.json")

if os.path.exists(result_json_src):
    shutil.copy(result_json_src, result_json_dst)
    print(f"📁 result.json copy in project: {result_json_dst}")
else:
    print("❌ Json result not found.")
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from prey_predator_Env_rllib import env as prey_predator_env
from agent_reward_logger import AgentRewardLogger
import os
import ray
import datetime
import shutil

# Starting the ray
ray.init(local_mode=True, include_dashboard=False)
os.environ["RLLIB_NUM_GPUS"] = "0"

# Configure the Env with render mode in None.
parallel_env = prey_predator_env(render_mode=None)
obs_space = parallel_env.observation_space("prey")
act_space = parallel_env.action_space("prey")

# Register environment
register_env("prey_predator_env", lambda config: ParallelPettingZooEnv(prey_predator_env(render_mode=None)))

# ✅ Define separate policies
policies = {
    "prey_policy": (
        None,
        obs_space,
        act_space,
        {}
    ),
    "predator_policy": (
        None,
        obs_space,
        act_space,
        {}
    )
}

# ✅ Policy mapping function
def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "prey":
        return "prey_policy"
    else:
        return "predator_policy"

# Training configuration
config_debug = {
    "rollout_fragment_length": 200,
    "env": "prey_predator_env",
    "framework": "torch",
    "callbacks": AgentRewardLogger,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "env_config": {},
    "num_workers": 0,
    "log_level": "ERROR",
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "tf_session_args": {
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
    }
}

print("\n🚀 Iniciando entrenamiento...")
results = tune.run(
    "PPO",
    stop={"episodes_total": 2000},
    config=config_debug,
    checkpoint_at_end=True,
    metric="episode_reward_mean",
    mode="max"
)

# Save the checkpoint, the model trained in that point
best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
checkpoint_path = best_trial.checkpoint.path

model_dir = os.path.join("PPO_Model_RL")
os.makedirs(model_dir, exist_ok=True)
model_checkpoint_path = os.path.join(model_dir, "best_model")
shutil.copytree(checkpoint_path, model_checkpoint_path, dirs_exist_ok=True)

with open("last_checkpoint.txt", "w") as f:
    f.write(model_checkpoint_path)

print("\n✅ TRAINING COMPLETED")
print(f"🔹 Checkpoint saved in: {model_checkpoint_path}")

# === Guardar result.json ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_dir = os.path.join("jsons")
os.makedirs(json_dir, exist_ok=True)

trial_dir = best_trial.logdir
result_json_src = os.path.join(trial_dir, "result.json")
result_json_dst = os.path.join(json_dir, f"result_{timestamp}.json")

if os.path.exists(result_json_src):
    shutil.copy(result_json_src, result_json_dst)
    print(f"📁 result.json copy in project: {result_json_dst}")
else:
    print("❌ Json result not found.")
