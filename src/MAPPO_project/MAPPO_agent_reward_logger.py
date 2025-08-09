import os
import json
from ray.rllib.algorithms.callbacks import DefaultCallbacks

"""
This file allow save the jsons with information for analysis.
- agent_rewards.json: Contains the rewards of each agent in each episode.
- termination_stats.json: Contains the number of times each agent was caught, trapped, or escaped
"""


class AgentRewardLogger(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_rewards = {"prey": [], "predator": []}

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.rewards_path = os.path.join(base_dir, "jsons", "agent_rewards.json")
        self.termination_path = os.path.join(base_dir, "jsons", "termination_stats.json")

        os.makedirs(os.path.dirname(self.rewards_path), exist_ok=True)
        print("🛠️ [DEBUG] Callback inicializado")
        print(f"🗂️ agent_rewards.json → {self.rewards_path}")
        print(f"🗂️ termination_stats.json → {self.termination_path}")

        # Create termination
        if not os.path.exists(self.termination_path):
            with open(self.termination_path, "w") as f:
                json.dump({"caught": 0, "trapped": 0, "escaped": 0}, f, indent=4)

    def on_episode_end(self, *, episode, **kwargs):
        # === Save rewards ===
        rewards = episode.agent_rewards
        prey_total = 0
        predator_total = 0

        for (agent_id, _), reward in rewards.items():
            if "prey" in agent_id:
                prey_total += reward
            elif "predator" in agent_id:
                predator_total += reward

        self.episode_rewards["prey"].append(prey_total)
        self.episode_rewards["predator"].append(predator_total)

        try:
            with open(self.rewards_path, "w") as f:
                json.dump(self.episode_rewards, f, indent=4)
        except Exception as e:
            print(f"❌ [ERROR] No se pudo guardar recompensas: {e}")

        try:
            cause = None
            for agent in ["prey", "predator"]:
                info = episode.last_info_for(agent)
                if info and "termination_reason" in info:
                    cause = info["termination_reason"]
                    break

            if cause is not None:
                with open(self.termination_path, "r") as f:
                    current_data = json.load(f)

                if cause in current_data:
                    current_data[cause] += 1
                else:
                    current_data[cause] = 1  

                with open(self.termination_path, "w") as f:
                    json.dump(current_data, f, indent=4)

            else:
                print(f"⚠️ [WARN] No se encontró termination_reason")

        except Exception as e:
            print(f"❌ [ERROR] No se pudo actualizar termination_stats: {e}")
