from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

from datetime import datetime

import pygame
import random
import math

import Environment.constant_variables as constant_variables
from Environment.Maze import Maze
from Environment.utils import load_prey_sprites, load_predator_sprites, load_tiles, initial_world_data, randomize_traps, generate_valid_start_positions
from Environment.Prey import Prey
from Environment.Predator import Predator

"""
This file has the own environment, created with PettingZoo API
Implements the neccesary thing for implement rllib algorithms for RL
"""

episode_rewards_log = {"prey": [], "predator": []}
class PreyPredatorEnv(ParallelEnv):
    
    metadata = {
    "render_modes": ["human"],
    "name": "predator_prey_v0",
    "is_parallelizable": True  # For MAPPO and wrapper
    }


    def __init__(self, render_mode=None):
        self.agents = ["predator", "prey"]
        self.possible_agents = self.agents[:]

        self.smell_evaporation_counter = 0

        self.render_mode = render_mode

        self.tile_size = constant_variables.tile_size
        self.grid_size = 10

        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32)
            for agent in self.agents
        }

        self.action_spaces = {
            agent: Discrete(4)
            for agent in self.agents
        }

        if self.render_mode == "human":
            self.animation_prey = load_prey_sprites()
            self.animation_predator = load_predator_sprites()
            self.tiles = load_tiles()
            self.window = None
            self.clock = None
        else:
            self.animation_prey = None
            self.animation_predator = None
            self.tiles = None
            self.window = None
            self.clock = None

        self.maze = Maze()
        self.initial_world_data = initial_world_data
        self.world_data = None
        self.agent_matrix = None
        self.state_matrix = None
        self.smell_matrix = None
        self.collision_rects = None

        self.prey = None
        self.predator = None
        self.pos = {}


    def _get_obs(self, agent):
        if agent == "prey":
            raw_obs = self.prey.obs_prey
        elif agent == "predator":
            raw_obs = self.predator.obs_predator
        else:
            raise ValueError(f"Unknown agent: {agent}")

        obs_np = self.normalize_obs_vector(raw_obs)

        # Padding just in case
        if obs_np.shape[0] < 30:
            obs_np = np.pad(obs_np, (0, 30 - obs_np.shape[0]), constant_values=0.0)

        return obs_np


    def normalize_obs_vector(self, raw_obs):
        normalized = []

        for i, val in enumerate(raw_obs):
            if i in [0, 1, 3, 4, 6, 7, 9, 10,
                    12, 13, 15, 16, 18, 19,
                    21, 22, 24, 25, 27, 28, 29]:  
                norm_val = (float(val) + 1.0) / 11.0
            elif i in [2, 5, 8, 11, 14, 17, 20, 23, 26]:
                norm_val = float(val) / 5.0
            elif i == 27:  # Booleano
                norm_val = float(val)
            elif i == 28:  # Intensidad de olor
                norm_val = max(float(val), 0.0) / 300.0
            else:
                norm_val = 0.0 

            normalized.append(min(norm_val, 1.0))  

        return np.array(normalized, dtype=np.float32)


    def reset(self, seed=None, options=None):
        self.agents = ["predator", "prey"]
        self.dones = {agent: False for agent in self.agents}
        self.current_step = 0 #Contador de pasos por episodio

        self.world_data = [row.copy() for row in self.initial_world_data]
        # Search valor positions for prey and predator
        positions = generate_valid_start_positions(self.world_data, tile_size=self.tile_size)

        # Update trap positions
        prey_start = (positions["prey_x_init"], positions["prey_y_init"])
        predator_start = (positions["predator_x_init"], positions["predator_y_init"])
        randomize_traps(self.world_data, prey_start, predator_start)

        self.state_matrix = [[0 for _ in range(10)] for _ in range(10)]
        self.agent_matrix = np.zeros((10, 10), dtype=np.uint8)
        self.smell_matrix = [[0 for _ in range(10)] for _ in range(10)]

        self.collision_rects = []
        for y, row in enumerate(self.world_data):
            for x, tile in enumerate(row):
                if tile in constant_variables.collision_tiles:
                    rect = pygame.Rect(
                        x * self.tile_size, y * self.tile_size,
                        self.tile_size, self.tile_size
                    )
                    self.collision_rects.append(rect)

        if self.render_mode == "human" and self.tiles is not None:
            self.maze.process_data(self.world_data, self.tiles)

        self._update_state_matrix_from_world()

        
        self.prey = Prey(positions["prey_px"], positions["prey_py"], self.animation_prey if self.animation_prey else [])
        self.predator = Predator(positions["predator_px"], positions["predator_py"], self.animation_predator if self.animation_predator else [])

        if self.render_mode == "human":
            self.prey.printHumanMode = True
            self.predator.printHumanMode = True

        self.agent_matrix[positions["prey_y_init"]][positions["prey_x_init"]] = 1
        self.agent_matrix[positions["predator_y_init"]][positions["predator_x_init"]] = 2

        self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                              self.prey.new_state_x, self.prey.new_state_y)
        self.predator.predator_sensor(self.state_matrix, self.agent_matrix,
                                      self.predator.new_state_x, self.predator.new_state_y)
        self.predator.smell(self.smell_matrix)
        self.predator.should_hunt()

        self.pos["prey"] = [positions["prey_x_init"], positions["prey_y_init"]]
        self.pos["predator"] = [positions["predator_x_init"], positions["predator_y_init"]]

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"position": self.pos[agent], "sensor_status": "active"} for agent in self.agents}

        return observations, infos


    def step(self, actions):

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"position": self.pos[agent]} for agent in self.agents}
        prey_falled = False

        delta_x_prey, delta_y_prey = self.action_to_delta(actions["prey"])
        delta_x_predator, delta_y_predator = self.action_to_delta(actions["predator"])
        
        # Prey movement ------------------------------------------------------------
        if self.prey.seen_predator:
            if delta_x_prey != 0 or delta_y_prey != 0:
                new_hitbox_prey = self.prey.hitbox().move(delta_x_prey, delta_y_prey)
                if not self.check_collision(new_hitbox_prey, self.collision_rects):
                    self.prey.movement(delta_x_prey, delta_y_prey)
                    self.smell_matrix[self.prey.new_state_y][self.prey.new_state_x] = constant_variables.smell_initial_strength

                    # Check collision with predator
                    if self.prey.hitbox().colliderect(self.predator.hitbox()):
                        rewards["predator"] += 10.0
                        rewards["prey"] -= 5.0
                        terminations["prey"] = True
                        terminations["predator"] = True
                    else:
                        rewards["prey"] += 0.1
                        rewards["prey"] += self.calculate_prey_dense_reward()
                        rewards["predator"] += self.calculate_predator_dense_rewards()
                    

                    self.update_agent_matrix(
                        self.agent_matrix,
                        self.prey.old_state_x, self.prey.old_state_y,
                        self.prey.new_state_x, self.prey.new_state_y
                    )

                    self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                                        self.prey.new_state_x, self.prey.new_state_y)
                    self.predator.predator_sensor(self.state_matrix, self.agent_matrix,
                                                self.predator.new_state_x, self.predator.new_state_y)
                    self.predator.smell(self.smell_matrix)
                    self.predator.should_hunt()
                    self.prey.fatigue += 1   

        # When a prey is over a trap
        prey_x = self.prey.hitbox().x // self.tile_size
        prey_y = self.prey.hitbox().y // self.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:

            if self.world_data[prey_y][prey_x] == 11:
                evasion_probability = self.prey.calculate_evasion_probability()
                if random.random() > evasion_probability:
                    terminations = {agent: True for agent in self.agents}
                    rewards["prey"] -= 5.0
                    prey_falled = True
        
                    
                    if self.render_mode == "human":
                        print(f"Prey fell in trap. Probability to avoid: {evasion_probability:.2f}")
                else:
                    if self.render_mode == "human":
                        print(f"Prey avoid the trap. Probability to avoid: {evasion_probability:.2f}")

        if delta_x_prey != 0 or delta_y_prey != 0:
            new_hitbox_prey = self.prey.hitbox().move(delta_x_prey, delta_y_prey)
            if not self.check_collision(new_hitbox_prey, self.collision_rects):
                self.prey.movement(delta_x_prey, delta_y_prey)
                self.smell_matrix[self.prey.new_state_y][self.prey.new_state_x] = constant_variables.smell_initial_strength

                # Prey Predator collision 
                if self.prey.hitbox().colliderect(self.predator.hitbox()):
                    rewards["predator"] += 10.0
                    rewards["prey"] -= 5.0
                    terminations["prey"] = True
                    terminations["predator"] = True
                else:
                    rewards["prey"] += 0.1
                    rewards["prey"] += self.calculate_prey_dense_reward()
                    rewards["predator"] += self.calculate_predator_dense_rewards()
                    

                self.update_agent_matrix(
                    self.agent_matrix,
                    self.prey.old_state_x, self.prey.old_state_y,
                    self.prey.new_state_x, self.prey.new_state_y
                )

                self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                                    self.prey.new_state_x, self.prey.new_state_y)
                self.predator.predator_sensor(self.state_matrix, self.agent_matrix,
                                            self.predator.new_state_x, self.predator.new_state_y)
                self.predator.smell(self.smell_matrix)
                self.predator.should_hunt()

        # When a prey is over a trap
        prey_x = self.prey.hitbox().x // self.tile_size
        prey_y = self.prey.hitbox().y // self.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:

            if self.world_data[prey_y][prey_x] == 11:
                evasion_probability = self.prey.calculate_evasion_probability()
                if random.random() > evasion_probability:
                    terminations = {agent: True for agent in self.agents}
                    rewards["prey"] -= 5.0
                    prey_falled = True
        
                    
                    if self.render_mode == "human":
                        print(f"Prey fell in trap. Probability to avoid: {evasion_probability:.2f}")
                else:
                    if self.render_mode == "human":
                        print(f"Prey avoid the trap. Probability to avoid: {evasion_probability:.2f}")


        # Predator movement ------------------------------------------------------------
        if self.predator.hunting: 

            if delta_x_predator != 0 or delta_y_predator != 0:
                new_hitbox_predator = self.predator.hitbox().move(delta_x_predator, delta_y_predator)
                if not self.check_collision(new_hitbox_predator, self.collision_rects):                
                    self.predator.movement(delta_x_predator, delta_y_predator)
                   
                   # Prey Predator collision 
                    if self.prey.hitbox().colliderect(self.predator.hitbox()):
                        rewards["predator"] += 10.0
                        rewards["prey"] -= 5.0
                        terminations["prey"] = True
                        terminations["predator"] = True
                    else:
                        rewards["prey"] += 0.1
                        rewards["prey"] += self.calculate_prey_dense_reward()
                        rewards["predator"] += self.calculate_predator_dense_rewards()
                    
                    self.update_agent_matrix(
                        self.agent_matrix,
                        self.predator.old_state_x, self.predator.old_state_y,
                        self.predator.new_state_x, self.predator.new_state_y
                    )

                self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                                    self.prey.new_state_x, self.prey.new_state_y)
                self.predator.predator_sensor(self.state_matrix, self.agent_matrix,
                                            self.predator.new_state_x, self.predator.new_state_y)
                self.predator.smell(self.smell_matrix)
                self.predator.should_hunt()

        if delta_x_predator != 0 or delta_y_predator != 0:
            new_hitbox_predator = self.predator.hitbox().move(delta_x_predator, delta_y_predator)
            if not self.check_collision(new_hitbox_predator, self.collision_rects):                
                self.predator.movement(delta_x_predator, delta_y_predator)
                   
                # Prey Predator collision 
                if self.prey.hitbox().colliderect(self.predator.hitbox()):
                    rewards["predator"] += 15.0
                    rewards["prey"] -= 10.0
                    terminations["prey"] = True
                    terminations["predator"] = True
                else:
                    rewards["prey"] += 0.1
                    rewards["prey"] += self.calculate_prey_dense_reward()
                    rewards["predator"] += self.calculate_predator_dense_rewards()
                    
                    self.update_agent_matrix(
                        self.agent_matrix,
                        self.predator.old_state_x, self.predator.old_state_y,
                        self.predator.new_state_x, self.predator.new_state_y
                    )

                self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                                    self.prey.new_state_x, self.prey.new_state_y)
                self.predator.predator_sensor(self.state_matrix, self.agent_matrix,
                                            self.predator.new_state_x, self.predator.new_state_y)
                self.predator.smell(self.smell_matrix)
                self.predator.should_hunt()
        
        # When predator is over a trap
        pred_x = self.predator.hitbox().x // self.tile_size
        pred_y = self.predator.hitbox().y // self.tile_size
        if 0 <= pred_x < 10 and 0 <= pred_y < 10:
            if self.world_data[pred_y][pred_x] == 11:
                rewards["predator"] -= 0.05  # Punish if predator crossing a trap
                
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        self.current_step += 1
        if self.current_step >= constant_variables.max_moves:
            truncations = {agent: True for agent in self.agents}
            rewards["prey"] += 1.0
                    
        self.smell_evaporation_counter += 1
        if self.smell_evaporation_counter >= constant_variables.smell_evaporation_interval:
            for y in range(10):
                for x in range(10):
                    if self.smell_matrix[y][x] > 0:
                        self.smell_matrix[y][x] -= 1
            self.smell_evaporation_counter = 0

        # Info of why finished the episode
        termination_reason = None
        if terminations["prey"] and terminations["predator"] and not prey_falled:
            termination_reason = "caught" 
        elif terminations["prey"] and terminations["predator"] and prey_falled:
            termination_reason = "trapped"
        elif all(truncations.values()):
            termination_reason = "escaped"

        for agent in self.agents:
            infos[agent]["termination_reason"] = termination_reason

        # Custom metrics for logging
        infos["__common__"] = {"termination_reason": termination_reason}


        return observations, rewards, terminations, truncations, infos


    def render(self, mode=None):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (constant_variables.width_windows, constant_variables.height_windows)
            )
            pygame.display.set_caption("Prey Predator")
            self.clock = pygame.time.Clock()

        self.window.fill(constant_variables.color_back)
        self._draw_grid()
        self.maze.draw(self.window)

        self.prey.update()
        self.predator.update()
        self.prey.draw(self.window, (255, 255, 0))
        self.predator.draw(self.window, (255, 255, 0))

        if hasattr(self, "game_over") and self.game_over:
            font = pygame.font.SysFont(None, 48)
            text = font.render("¡Has perdido!", True, (255, 0, 0))
            self.window.blit(text, (50, constant_variables.height_windows // 2 - 20))

        pygame.display.update()
        self.clock.tick(constant_variables.FPS)


    def calculate_predator_dense_rewards(self):
        reward = 0.0

        # === Predator position ===
        pred_x = self.predator.new_state_x
        pred_y = self.predator.new_state_y
        pred_old_x = self.predator.old_state_x
        pred_old_y = self.predator.old_state_y

        # === Punish by stay still ===
        if pred_x == pred_old_x and pred_y == pred_old_y:
            reward -= 0.05  # 

        # === If see the prey ===
        if self.predator.seen_prey:
            prey_x = self.prey.new_state_x
            prey_y = self.prey.new_state_y

            dist_now = math.hypot(prey_x - pred_x, prey_y - pred_y)
            dist_before = math.hypot(prey_x - pred_old_x, prey_y - pred_old_y)

            delta = dist_before - dist_now

            if delta > 0:
                reward += 0.1  # Approach
            elif delta < 0:
                reward -= 0.05  # Away

        # === If NOT see the prey ===
        else:
            if pred_x != pred_old_x or pred_y != pred_old_y:
                reward += 0.05  # Moving and exploring

        return reward


    def calculate_prey_dense_reward(self):
        reward = 0.0

        prey_x = self.prey.new_state_x
        prey_y = self.prey.new_state_y
        prey_old_x = self.prey.old_state_x
        prey_old_y = self.prey.old_state_y

        pred_x = self.predator.new_state_x
        pred_y = self.predator.new_state_y

        moved = not (prey_x == prey_old_x and prey_y == prey_old_y)

        if self.prey.seen_predator:
            dist_now = math.hypot(pred_x - prey_x, pred_y - prey_y)
            dist_before = math.hypot(pred_x - prey_old_x, pred_y - prey_old_y)

            delta = dist_now - dist_before

            if delta > 0:
                reward += 0.1  # Away
            elif delta < 0:
                reward -= 0.05  # Approach

        else:

            if moved:
                reward += 0.05  # Exploring
            else:
                reward -= 0.05  # Not moving

        return reward


    def action_to_delta(self, action):
        if action == 0:
            return 0, -50
        elif action == 1:
            return 0, 50
        elif action == 2:
            return -50, 0
        elif action == 3:
            return 50, 0
        return 0, 0


    def check_collision(self, agent_rect, collision_rects):
        # Walls collision
        for rect in collision_rects:
            if agent_rect.colliderect(rect):
                return True
        
        # Border collision
        if not pygame.Rect(0, 0, constant_variables.width_windows, constant_variables.height_windows).contains(agent_rect):
            return True

        return False


    def update_agent_matrix(self, agent_matrix, x, y, new_x, new_y):
        val = agent_matrix[y][x]
        agent_matrix[y][x] = 0
        agent_matrix[new_y][new_x] = val


    def _update_state_matrix_from_world(self):
        for y in range(len(self.world_data)):
            for x in range(len(self.world_data[y])):
                tile = self.world_data[y][x]
                if tile in [0, 1, 2, 3, 4, 5, 6, 7]:
                    self.state_matrix[y][x] = 3
                elif tile in [8, 9, 10]:
                    self.state_matrix[y][x] = 5
                elif tile == 11:
                    self.state_matrix[y][x] = 4


    def _draw_grid(self):
            for x in range(10):
                pygame.draw.line(
                    self.window, (203, 50, 52),
                    (x * self.tile_size, 0),
                    (x * self.tile_size, constant_variables.height_windows)
                )
                pygame.draw.line(
                    self.window, (203, 50, 52),
                    (0, x * self.tile_size),
                    (constant_variables.width_windows, x * self.tile_size)
                )


    def observation_space(self, agent):
        return self.observation_spaces[agent]


    def action_space(self, agent):
        return self.action_spaces[agent]
    
    
    def close(self):
        if hasattr(self, "window") and self.window is not None:
            pygame.quit()
            self.window = None