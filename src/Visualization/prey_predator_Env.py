# prey_predator_Env
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

import datetime
import os
import json
from datetime import datetime

import pygame
import random
import math

import constant_variables
from Maze import Maze
from utils import load_prey_sprites, load_predator_sprites, load_tiles, initial_world_data, randomize_traps
from Prey import Prey
from Predator import Predator

"""
This file has the own environment, created with PettingZoo API
Implements the neccesary thing for implement rllib algorithms for RL
"""



episode_rewards_log = {"prey": [], "predator": []}
class PreyPredatorEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "predator_prey_v0"}

    def __init__(self, render_mode=None):
        self.agents = ["predator", "prey"]
        self.possible_agents = self.agents[:]

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

        # Padding por si acaso
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
        randomize_traps(self.world_data)

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

        
        self.prey = Prey(25, 475, self.animation_prey if self.animation_prey else [])
        self.predator = Predator(475, 25, self.animation_predator if self.animation_predator else [])
        
        if self.render_mode == "human":
            self.prey.printHumanMode = True
            self.predator.printHumanMode = True

        self.agent_matrix[9][0] = 1
        self.agent_matrix[0][9] = 2

        self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                              self.prey.new_state_x, self.prey.new_state_y)
        self.predator.activate_sensor(self.state_matrix, self.agent_matrix,
                                      self.predator.new_state_x, self.predator.new_state_y)
        self.predator.smell(self.smell_matrix)
        self.predator.should_hunt()

        self.pos["prey"] = [9, 0]
        self.pos["predator"] = [0, 9]

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"position": self.pos[agent], "sensor_status": "active"} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        delta_x_prey, delta_y_prey = self.action_to_delta(actions["prey"])
        delta_x_predator, delta_y_predator = self.action_to_delta(actions["predator"])

        # Movimiento de la presa
        if delta_x_prey != 0 or delta_y_prey != 0:
            new_hitbox_prey = self.prey.hitbox().move(delta_x_prey, delta_y_prey)
            if not self.check_collision(new_hitbox_prey, self.collision_rects):
                self.prey.movement(delta_x_prey, delta_y_prey)
                self.smell_matrix[self.prey.new_state_y][self.prey.new_state_x] = constant_variables.smell_initial_strength

                self.update_agent_matrix(
                    self.agent_matrix,
                    self.prey.old_state_x, self.prey.old_state_y,
                    self.prey.new_state_x, self.prey.new_state_y
                )

                self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                                    self.prey.new_state_x, self.prey.new_state_y)
                self.predator.activate_sensor(self.state_matrix, self.agent_matrix,
                                            self.predator.new_state_x, self.predator.new_state_y)
                self.predator.smell(self.smell_matrix)
                self.predator.should_hunt()

        # Movimiento del depredador (simplificado para compatibilidad RLlib)
        collision_x, collision_y = delta_x_predator, delta_y_predator

        if delta_x_predator != 0 or delta_y_predator != 0:

            if self.predator.hunting:
                if delta_x_predator != 0:
                    collision_x += constant_variables.speed_increase * (1 if delta_x_predator > 0 else -1)
                if delta_y_predator != 0:
                    collision_y += constant_variables.speed_increase * (1 if delta_y_predator > 0 else -1)

            collision, distance, _ = self.predator.check_collision_not_zero(
                self.predator.hitbox(), self.predator.hitbox().move(collision_x, collision_y), self.collision_rects
            )

            border, border_dist, _ = self.predator.check_border_collision(
                self.predator.hitbox(), self.predator.hitbox().move(collision_x, collision_y)
            )

            if not border and not collision:
                self.predator.movement(delta_x_predator, delta_y_predator)
                if self.predator.move:
                    if self.predator.hunting:
                        self._update_agent_matrix_fast(self.agent_matrix)
                    elif self.predator.adjusting:
                        self._update_agent_matrix_adjust(self.agent_matrix)
                    else:
                        self.predator.transition_normal_fast()
                        self.update_agent_matrix(
                            self.agent_matrix,
                            self.predator.old_state_x, self.predator.old_state_y,
                            self.predator.new_state_x, self.predator.new_state_y
                        )
                    self.prey.prey_sensor(
                        self.state_matrix, self.agent_matrix,
                        self.prey.new_state_x, self.prey.new_state_y
                    )
                    self.predator.activate_sensor(
                        self.state_matrix, self.agent_matrix,
                        self.predator.new_state_x, self.predator.new_state_y
                    )
                    self.predator.smell(self.smell_matrix)

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"position": self.pos[agent]} for agent in self.agents}

        # Prey Predator collision 
        if self.prey.hitbox().colliderect(self.predator.hitbox()):
            rewards["predator"] += 10.0
            rewards["prey"] -= 5.0
            terminations["prey"] = True
            terminations["predator"] = True
        else:
            #  rewards["prey"] += 0.1
            if self.prey.old_state_x == self.prey.new_state_x and self.prey.old_state_y == self.prey.new_state_y:
                rewards["prey"] -= 0.5
            rewards["predator"] -= 0.05
            rewards["predator"] += self.calculate_predator_dense_rewards()

        # WHen a prey is over a trap
        prey_x = self.prey.hitbox().x // self.tile_size
        prey_y = self.prey.hitbox().y // self.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:

            if self.world_data[prey_y][prey_x] == 11:
                evasion_probability = self.prey.calculate_evasion_probability()
                if random.random() > evasion_probability:
                    terminations = {agent: True for agent in self.agents}
                    rewards["prey"] -= 5.0

        
                    
                    if self.render_mode == "human":
                        print(f"Prey fell in trap. Probability to avoid: {evasion_probability:.2f}")
                else:
                    if self.render_mode == "human":
                        print(f"Prey avoid the trap. Probability to avoid: {evasion_probability:.2f}")

        observations = {agent: self._get_obs(agent) for agent in self.agents}


        self.current_step += 1
        if self.current_step >= constant_variables.max_moves:
            truncations = {agent: True for agent in self.agents}
            if not terminations["prey"]:
                rewards["prey"] += 10.0
                rewards["predator"] -= 5.0

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
        reward = 0

        if self.predator.hunting:
            pred_x = self.predator.hunt1_new_state_x
            pred_y = self.predator.hunt1_new_state_y
            pred_old_x = self.predator.hunt1_old_state_x
            pred_old_y = self.predator.hunt1_old_state_y
        else:
            pred_x = self.predator.new_state_x
            pred_y = self.predator.new_state_y
            pred_old_x = self.predator.old_state_x
            pred_old_y = self.predator.old_state_y

        if pred_x == pred_old_x and pred_y == pred_old_y:
            reward -= 0.1
        
        if self.predator.seen_prey:
            prey_x = self.prey.new_state_x
            prey_y = self.prey.new_state_y

            dist_now = math.hypot(prey_x - pred_x, prey_y - pred_y)
            dist_before = math.hypot(prey_x - pred_old_x, prey_y - pred_old_y)

            delta = dist_before - dist_now
            if delta > 0:
                reward += delta * 0.5
            elif delta < 0:
                reward += delta * 0.2  # penalización leve si se aleja
        else:
            reward += 0.1  # recompensa leve si se acerca
        
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
        for rect in collision_rects:
            if agent_rect.colliderect(rect):
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

    def _update_agent_matrix_fast(self, agent_matrix):
            predator = self.predator  # para no escribir self.predator. todo el tiempo

            agent_matrix[predator.new_state_y][predator.new_state_x] = 0 
            agent_matrix[predator.hunt1_old_state_y][predator.hunt1_old_state_x] = 0
            agent_matrix[predator.hunt2_old_state_y][predator.hunt2_old_state_x] = 0

            agent_matrix[predator.hunt1_new_state_y][predator.hunt1_new_state_x] = 2
            agent_matrix[predator.hunt2_new_state_y][predator.hunt2_new_state_x] = 2
    
    def _update_agent_matrix_adjust(self, agent_matrix):
        predator = self.predator

        agent_matrix[predator.hunt1_new_state_y][predator.hunt1_new_state_x] = 0
        agent_matrix[predator.hunt2_new_state_y][predator.hunt2_new_state_x] = 0
        agent_matrix[predator.new_state_y][predator.new_state_x] = 2

        predator.adjusting = False

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