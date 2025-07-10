from ray.rllib.algorithms.callbacks import DefaultCallbacks
import os
import json

class AgentRewardLogger(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # Ruta RELATIVA para compatibilidad con tu proyecto
        self.rewards_path = os.path.join(os.path.dirname(__file__), "../../jsons/agent_rewards.json")
        self.rewards_path = os.path.abspath(self.rewards_path)
        os.makedirs(os.path.dirname(self.rewards_path), exist_ok=True)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        #print("📌 Callback: End episode")

        # Create json file for rewards
        if not os.path.exists(self.rewards_path):
            with open(self.rewards_path, "w") as f:
                json.dump({"prey": [], "predator": []}, f)

        # Charge the dates
        with open(self.rewards_path, "r") as f:
            rewards_data = json.load(f)

        # Calculate rewards
        episode_rewards = {"prey": 0.0, "predator": 0.0}
        for (agent_id, _), reward in episode.agent_rewards.items():
            if agent_id in episode_rewards:
                episode_rewards[agent_id] += reward

        # Add and save
        rewards_data["prey"].append(episode_rewards["prey"])
        rewards_data["predator"].append(episode_rewards["predator"])

        with open(self.rewards_path, "w") as f:
            json.dump(rewards_data, f, indent=4)

        #print(f"📁 Callback: Recompensas guardadas en: {self.rewards_path}")