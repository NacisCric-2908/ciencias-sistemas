from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import numpy as np
import constant_variables
from Maze import Maze
from utils import load_prey_sprites, load_predator_sprites, load_tiles, initial_world_data, randomize_traps
import random

class PreyPredatorEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "predator_prey_v0"}

    def __init__(self, render_mode=None):
        self.agents = ["predator", "prey"]
        self.render_mode = render_mode
        self.pos = {}  # Posiciones actualizadas por reset

        self.window = None
        self.clock = None

        # Parámetros fijos del entorno
        self.grid_size = 10
        self.tile_size = constant_variables.tile_size

        # Espacios de acción (4 direcciones)
        self.action_spaces = {
            agent: spaces.Discrete(4)
            for agent in self.agents
        }

        # Observación como matriz combinada: state + agent
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=5, shape=(10, 10, 3), dtype=np.uint8)
            for agent in self.agents
        }

        # Recursos gráficos (si renderizas)
        self.animation_prey = load_prey_sprites()
        self.animation_predator = load_predator_sprites()
        self.tiles = load_tiles()
        self.maze = Maze()

        # Estado general
        self.initial_world_data = initial_world_data

        # Esto se inicializa en reset()
        self.world_data = None
        self.collision_rects = None
        self.state_matrix = None
        self.agent_matrix = None
        self.smell_matrix = None
        self.prey = None
        self.predator = None

    def _get_obs(self):
        return np.stack([
            self.state_matrix,
            self.agent_matrix,
            self.smell_matrix
        ], axis=-1)

    def reset(self, seed=None, options=None):
        self.agents = ["predator", "prey"]

        # Copiar y preparar el mundo
        self.world_data = [row.copy() for row in self.initial_world_data]
        randomize_traps(self.world_data)

        # Colisiones
        self.collision_rects = []
        for y, row in enumerate(self.world_data):
            for x, tile in enumerate(row):
                if tile in constant_variables.collision_tiles:
                    rect = pygame.Rect(
                        x * self.tile_size, y * self.tile_size,
                        self.tile_size, self.tile_size
                    )
                    self.collision_rects.append(rect)

        # Laberinto
        self.maze.process_data(self.world_data, self.tiles)

        # Matrices
        self.state_matrix = [[0 for _ in range(10)] for _ in range(10)]
        self.agent_matrix = np.zeros((10, 10), dtype=np.uint8)
        self.smell_matrix = [[0 for _ in range(10)] for _ in range(10)]

        self._update_state_matrix_from_world()

        # Crear agentes (objetos)
        from Prey import Prey
        from Predator import Predator
        self.prey = Prey(25, 475, self.animation_prey)
        self.predator = Predator(475, 25, self.animation_predator)

        # Actualizar matrices
        self.agent_matrix[9][0] = 1  # Prey
        self.agent_matrix[0][9] = 2  # Predator

        # Activar sensores
        self.prey.prey_sensor(self.state_matrix, self.agent_matrix,
                              self.prey.new_state_x, self.prey.new_state_y)
        self.predator.activate_sensor(self.state_matrix, self.agent_matrix,
                                      self.predator.new_state_x, self.predator.new_state_y)
        self.predator.smell(self.smell_matrix)

        # Guardar posiciones
        self.pos["prey"] = [9, 0]
        self.pos["predator"] = [0, 9]

        # Retornar observaciones de ambos agentes
        obs = {
            agent: self._get_obs()
            for agent in self.agents
        }
        return obs

    def step(self, actions):
        delta_x_prey, delta_y_prey = self.action_to_delta(actions["prey"])
        delta_x_predator, delta_y_predator = self.action_to_delta(actions["predator"])

        # =========================
        # MOVIMIENTO DE LA PRESA
        # =========================
        if not self.check_collision(self.prey.hitbox().move(delta_x_prey, delta_y_prey), self.collision_rects):
            self.prey.movement(delta_x_prey, delta_y_prey)
            self.smell_matrix[self.prey.new_state_y][self.prey.new_state_x] = constant_variables.smell_initial_strength

            if delta_x_prey != 0 or delta_y_prey != 0:
                self.update_agent_matrix(
                    self.agent_matrix,
                    self.prey.old_state_x, self.prey.old_state_y,
                    self.prey.new_state_x, self.prey.new_state_y
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

        # =========================
        # MOVIMIENTO DEL DEPREDADOR
        # =========================
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
            # (Aquí puedes agregar los casos de colisión parcial si lo deseas)

        # =========================
        # TERMINACIÓN DEL EPISODIO
        # =========================
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Fin si el depredador atrapa a la presa
        if self.prey.hitbox().colliderect(self.predator.hitbox()):
            terminations = {agent: True for agent in self.agents}
            rewards = {"predator": 1.0, "prey": -1.0}

        # Fin si la presa pisa una trampa
        prey_x = self.prey.hitbox().x // self.tile_size
        prey_y = self.prey.hitbox().y // self.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:

            if self.world_data[prey_y][prey_x] == 11:
                evasion_probability = self.prey.calculate_evasion_probability()
                if random.random() > evasion_probability:
                    terminations = {agent: True for agent in self.agents}
                    rewards = {"predator": 1.0, "prey": -1.0}
                    print(f"La presa cayó en la trampa. Probabilidad de evadir: {evasion_probability:.2f}")
                else:
                    print(f"La presa EVADIÓ la trampa. Probabilidad de evadir: {evasion_probability:.2f}")


        observations = {
            agent: self._get_obs()
            for agent in self.agents
        }

        # =========================
        # EVAPORACIÓN DEL OLOR
        # =========================
        constant_variables.smell_evaporation_counter += 1
        if constant_variables.smell_evaporation_counter >= constant_variables.smell_evaporation_interval:
            for y in range(10):
                for x in range(10):
                    if self.smell_matrix[y][x] > 0:
                        self.smell_matrix[y][x] -= 1
            constant_variables.smell_evaporation_counter = 0

            

        return observations, rewards, terminations, truncations, infos


    def action_to_delta(self, action):
        # 0: up, 1: down, 2: left, 3: right
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


    def update_agent_matrix(self, agent_matrix, x, y, new_x, new_y):
        #Change states of the prey in the agent matrix
        aux_value = agent_matrix[y][x] 
        agent_matrix[y][x] = agent_matrix[new_y][new_x]   
        agent_matrix[new_y][new_x] = aux_value 

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


    def print_state_matrix(self, state_matrix): 
        print("Matriz de estado:")
        for row in state_matrix:
            print(row)
        print("-" * 40)

    def print_agent_matrix(self, agent_matrix):
        print("Matriz de agentes:")
        for row in agent_matrix:
            print(row)
        print("-" * 40)

    def print_smell_matrix(smell_matrix):
        print("Matriz de olor:")
        for row in smell_matrix:
            print(row)
        print("-" * 40)

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((constant_variables.width_windows, constant_variables.height_windows))
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

    def close(self):
        if hasattr(self, "window") and self.window is not None:
            pygame.quit()
            self.window = None


