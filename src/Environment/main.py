#This file is used for prove the features

import pygame

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Environment.Agent import Agent
from Environment.Maze import Maze
from Environment.Prey import Prey
from Environment.Predator import Predator
import Environment.constant_variables as constant_variables
import numpy as np
import random
import csv
import os
import math

# Start the game
pygame.init()

# Show main window (width, height)
window = pygame.display.set_mode((constant_variables.width_windows, constant_variables.height_windows))


rewards_prey = 0.0
rewards_predator = 0.0

# Put a name
pygame.display.set_caption("Prey Predator")

# Prey animation
animation_prey = []
for i in range(7):
    img_prey = pygame.image.load(f"src/assets/images/prey/Prey_{i}.png")
    img_prey = pygame.transform.scale(img_prey, (50, 50))
    animation_prey.append(img_prey)

# Predator animation
animation_predator = []
for i in range(7):
    img_predator = pygame.image.load(f"src/assets/images/predator/Predator_{i}.png")
    img_predator = pygame.transform.scale(img_predator, (50, 50))
    animation_predator.append(img_predator)

# Tile images
tile_list = []
for x in range(12):
    tile_image = pygame.image.load(f"src/assets/images/tiles/Tile ({x+1}).png")
    tile_image = pygame.transform.scale(tile_image, (constant_variables.tile_size, constant_variables.tile_size))
    tile_list.append(tile_image)

# World data
initial_world_data = [
    [11, 7, 8, 9, 10, 9, 10, 8, 10, 9],
    [9, 4, 4, 11, 7, 10, 4, 4, 4, 8],
    [8, 4, 4, 9, 6, 9, 5, 11, 5, 10],
    [10, 6, 8, 9, 10, 9, 5, 8, 5, 9],
    [8, 9, 10, 0, 4, 9, 6, 8, 6, 10],
    [10, 7, 8, 9, 4, 9, 10, 8, 10, 9],
    [8, 5, 11, 9, 2, 3, 3, 2, 1, 8],
    [9, 5, 8, 9, 10, 9, 10, 8, 10, 9],
    [10, 6, 10, 0, 2, 3, 1, 9, 0, 3],
    [10, 9, 8, 9, 10, 9, 10, 8, 10, 11]
]

# agregar
def flatten_matrix(matrix, prefix):
    """Convierte una matriz 10x10 en un diccionario con claves prefijadas."""
    flattened = {}
    for y in range(10):
        for x in range(10):
            flattened[f"{prefix}_{y}_{x}"] = matrix[y][x]
    return flattened

# agregar
def save_to_csv(data, filename="game_data.csv"):
    """Guarda los datos en un archivo CSV."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()  # Solo escribe el encabezado si el archivo no existe
        writer.writerow(data)
        
        
def generate_valid_start_positions(world_data, tile_size=50, min_distance=6):
    valid_positions = []
    for y, row in enumerate(world_data):
        for x, tile in enumerate(row):
            if tile in [8, 9, 10]:  # Caminos válidos
                valid_positions.append((x, y))

    max_attempts = 1000
    for _ in range(max_attempts):
        prey_pos = random.choice(valid_positions)
        predator_pos = random.choice(valid_positions)
        if prey_pos != predator_pos:
            dist = abs(prey_pos[0] - predator_pos[0]) + abs(prey_pos[1] - predator_pos[1])
            if dist >= min_distance:
                prey_px = prey_pos[0] * tile_size + 25
                prey_py = prey_pos[1] * tile_size + 25
                predator_px = predator_pos[0] * tile_size + 25
                predator_py = predator_pos[1] * tile_size + 25
                return (
                    (prey_px, prey_py, prey_pos[0], prey_pos[1]),
                    (predator_px, predator_py, predator_pos[0], predator_pos[1])
                )

    raise ValueError("No se pudieron asignar posiciones iniciales válidas con la distancia mínima requerida.")


def randomize_traps(world_data, prey_start, predator_start, min_distance=2):
    trap_positions = []
    path_positions = []

    for y, row in enumerate(world_data):
        for x, tile in enumerate(row):
            if tile == 11:
                trap_positions.append((x, y))
            elif tile in [8, 9, 10]:
                pos = (x, y)
                if (
                    abs(pos[0] - prey_start[0]) > min_distance or
                    abs(pos[1] - prey_start[1]) > min_distance
                ) and (
                    abs(pos[0] - predator_start[0]) > min_distance or
                    abs(pos[1] - predator_start[1]) > min_distance
                ):
                    path_positions.append(pos)

    for x, y in trap_positions:
        world_data[y][x] = random.choice([8, 9, 10])

    random.shuffle(path_positions)
    new_trap_positions = path_positions[:len(trap_positions)]
    for x, y in new_trap_positions:
        world_data[y][x] = 11

    return world_data


def create_game():
    world_data = [row.copy() for row in initial_world_data]

    # 🧠 Generar posiciones iniciales válidas
    (prey_px, prey_py, prey_x_init, prey_y_init), (pred_px, pred_py, pred_x_init, pred_y_init) = generate_valid_start_positions(world_data)

    # 🪤 Randomizar trampas lejos de las posiciones iniciales
    randomize_traps(world_data, (prey_x_init, prey_y_init), (pred_x_init, pred_y_init))

    # 🧱 Colisiones
    collision_rects = []
    for y, row in enumerate(world_data):
        for x, tile in enumerate(row):
            if tile in constant_variables.collision_tiles:
                rect = pygame.Rect(x * constant_variables.tile_size, y * constant_variables.tile_size,
                                   constant_variables.tile_size, constant_variables.tile_size)
                collision_rects.append(rect)

    maze = Maze()
    maze.process_data(world_data, tile_list)

    state_matrix = [[0 for _ in range(10)] for _ in range(10)]
    agent_matrix = np.zeros_like(state_matrix)
    smell_matrix = [[0 for _ in range(10)] for _ in range(10)]

    update_state_matrix_from_world(world_data, state_matrix)
    agent_matrix[prey_y_init][prey_x_init] = 1
    agent_matrix[pred_y_init][pred_x_init] = 2

    prey1 = Prey(prey_px, prey_py, animation_prey)
    predator1 = Predator(pred_px, pred_py, animation_predator)

    return world_data, collision_rects, maze, state_matrix, agent_matrix, prey1, predator1, smell_matrix


def update_state_matrix_from_world(world_data, state_matrix):
    for y in range(len(world_data)):
        for x in range(len(world_data[y])):
            tile = world_data[y][x]
            if tile in [0,1,2,3,4,5,6,7]:
                state_matrix[y][x] = 3
            elif tile in [8,9,10]:
                state_matrix[y][x] = 5
            elif tile == 11:
                state_matrix[y][x] = 4

def update_agent_matrix(agent_matrix, x, y, new_x, new_y):
    #Change states of the prey in the agent matrix
    aux_value = agent_matrix[y][x] 
    agent_matrix[y][x] = agent_matrix[new_y][new_x]   
    agent_matrix[new_y][new_x] = aux_value 

def print_state_matrix(state_matrix): 
    print("Matriz de estado:")
    for row in state_matrix:
        print(row)
    print("-" * 40)

def print_agent_matrix(agent_matrix):
    print("Matriz de agentes:")
    for row in agent_matrix:
        print(row)
    print("-" * 40)

def print_smell_matrix(smell_matrix):
    print("Matriz de olor:")
    for row in smell_matrix:
        print(row)
    print("-" * 40)
    
def draw_grid():
    for x in range(10):
        pygame.draw.line(window, (203, 50, 52), (x * constant_variables.tile_size, 0),
                         (x * constant_variables.tile_size, constant_variables.height_windows))
        pygame.draw.line(window, (203, 50, 52), (0, x * constant_variables.tile_size),
                         (constant_variables.width_windows, x * constant_variables.tile_size))

def check_collision(agent_rect, collision_rects):
    # Colisión con paredes
    for rect in collision_rects:
        if agent_rect.colliderect(rect):
            return True
    
    # Colisión con bordes de la ventana
    if not pygame.Rect(0, 0, constant_variables.width_windows, constant_variables.height_windows).contains(agent_rect):
        return True

    return False




def calculate_predator_dense_rewards():
        reward = 0.0

        pred_x = predator1.new_state_x
        pred_y = predator1.new_state_y
        pred_old_x = predator1.old_state_x
        pred_old_y = predator1.old_state_y

        # === Penalizacion si se queda quieto ===
        if pred_x == pred_old_x and pred_y == pred_old_y:
            reward -= 0.05  # castigo por no moverse

        # === Si ve a la presa ===
        if predator1.seen_prey:
            prey_x = prey1.new_state_x
            prey_y = prey1.new_state_y

            dist_now = math.hypot(prey_x - pred_x, prey_y - pred_y)
            dist_before = math.hypot(prey_x - pred_old_x, prey_y - pred_old_y)

            delta = dist_before - dist_now

            if delta > 0:
                reward += 0.1  # se acercó
            elif delta < 0:
                reward -= 0.05  # se alejó

        # === Si NO ve a la presa ===
        else:
            if pred_x != pred_old_x or pred_y != pred_old_y:
                reward += 0.05  # se está moviendo y explorando

        return reward


def calculate_prey_dense_reward():
        reward = 0.0

        prey_x = prey1.new_state_x
        prey_y = prey1.new_state_y
        prey_old_x = prey1.old_state_x
        prey_old_y = prey1.old_state_y

        pred_x = predator1.new_state_x
        pred_y = predator1.new_state_y

        moved = not (prey_x == prey_old_x and prey_y == prey_old_y)

        if prey1.seen_predator:
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


# Initialize game
world_data, collision_rects, maze, state_matrix, agent_matrix, prey1, predator1, smell_matrix = create_game()
prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
predator1.predator_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
predator1.smell(smell_matrix)

# Initialize movement flags
move_right_prey = False
move_left_prey = False
move_up_prey = False
move_down_prey = False
move_right_predator = False
move_left_predator = False
move_up_predator = False
move_down_predator = False

clock = pygame.time.Clock()
run = True
game_over = False

###################################################
print_state_matrix(state_matrix)
print_agent_matrix(agent_matrix)
print_smell_matrix(smell_matrix)
###################################################

while run:
    clock.tick(constant_variables.FPS)
    window.fill(constant_variables.color_back)
    draw_grid()
    
    # Smell agregar ------
    constant_variables.smell_evaporation_counter += 1
    if constant_variables.smell_evaporation_counter >= constant_variables.smell_evaporation_interval:
        for y in range(10):
            for x in range(10):
                if smell_matrix[y][x] > 0:
                    smell_matrix[y][x] -= 1
        constant_variables.smell_evaporation_counter = 0
    #-------------------------    

    # Draw maze
    maze.draw(window)

    if not game_over:
        delta_x_prey = delta_y_prey = delta_x_predator = delta_y_predator = 0
        if move_right_prey: delta_x_prey = constant_variables.speed_prey
        if move_left_prey: delta_x_prey = -constant_variables.speed_prey
        if move_up_prey: delta_y_prey = -constant_variables.speed_prey
        if move_down_prey: delta_y_prey = constant_variables.speed_prey
        if move_right_predator: delta_x_predator = constant_variables.speed_predator
        if move_left_predator: delta_x_predator = -constant_variables.speed_predator
        if move_up_predator: delta_y_predator = -constant_variables.speed_predator
        if move_down_predator: delta_y_predator = constant_variables.speed_predator

        # Move prey

        if prey1.seen_predator:
            if delta_x_prey != 0 or delta_y_prey != 0:
                if not check_collision(prey1.hitbox().move(delta_x_prey, delta_y_prey), collision_rects): #Not move in collisions
                    prey1.movement(delta_x_prey, delta_y_prey)
                    smell_matrix[prey1.new_state_y][prey1.new_state_x] = constant_variables.smell_initial_strength
                    update_agent_matrix(agent_matrix, prey1.old_state_x, prey1.old_state_y, prey1.new_state_x, prey1.new_state_y)
                    prey1.fatigue += 1        
            
        if delta_x_prey != 0 or delta_y_prey != 0:
            if not check_collision(prey1.hitbox().move(delta_x_prey, delta_y_prey), collision_rects): #Not move in collisions
                prey1.movement(delta_x_prey, delta_y_prey)
                smell_matrix[prey1.new_state_y][prey1.new_state_x] = constant_variables.smell_initial_strength
                update_agent_matrix(agent_matrix, prey1.old_state_x, prey1.old_state_y, prey1.new_state_x, prey1.new_state_y)
                prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                predator1.predator_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                predator1.smell(smell_matrix)
                predator1.should_hunt()  # Check if predator should hunt based on ML model
                print_state_matrix(state_matrix)
                print_agent_matrix(agent_matrix)
                print_smell_matrix(smell_matrix)
                print("Reward Prey:", rewards_prey)
                print("Reward Predator:", rewards_predator)
                    
        # Move predator 

        if predator1.hunting:
            if delta_x_predator != 0 or delta_y_predator != 0:
                # Check collision with walls
                if not check_collision(predator1.hitbox().move(delta_x_predator, delta_y_predator), collision_rects): #Not move in collisions
                        if not check_collision(predator1.hitbox().move(delta_x_predator, delta_y_predator), collision_rects): #Not move in collisions
                            predator1.movement(delta_x_predator, delta_y_predator)
                            # Check collision with predator
                            if prey1.hitbox().colliderect(predator1.hitbox()):
                                rewards_predator += 10.0
                                rewards_prey -= 5.0
                                game_over = True

                            else:
                                rewards_prey+= 0.1
                                rewards_prey+= calculate_prey_dense_reward()
                                rewards_predator += calculate_predator_dense_rewards()
                            update_agent_matrix(agent_matrix, predator1.old_state_x, predator1.old_state_y, predator1.new_state_x, predator1.new_state_y)

        if delta_x_predator != 0 or delta_y_predator != 0:
            if not check_collision(predator1.hitbox().move(delta_x_predator, delta_y_predator), collision_rects): #Not move in collisions
                predator1.movement(delta_x_predator, delta_y_predator)
                # Check collision with predator
                if prey1.hitbox().colliderect(predator1.hitbox()):
                    rewards_predator += 10.0
                    rewards_prey -= 5.0
                    game_over = True

                else:
                    rewards_prey+= 0.1
                    rewards_prey+= calculate_prey_dense_reward()
                    rewards_predator += calculate_predator_dense_rewards()

                update_agent_matrix(agent_matrix, predator1.old_state_x, predator1.old_state_y, predator1.new_state_x, predator1.new_state_y)
                predator1.predator_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                predator1.smell(smell_matrix)
                predator1.should_hunt()  # Check if predator should hunt based on ML model
                print_state_matrix(state_matrix)
                print_agent_matrix(agent_matrix)
                print_smell_matrix(smell_matrix)

                print("Hunting mode:", predator1.hunting)
                print("Seen: ", predator1.seen_prey)
                print("Smell: ", predator1.smell_intensity)  

                print("Reward Prey:", rewards_prey)
                print("Reward Predator:", rewards_predator)


        # Check collision with predator
        if prey1.hitbox().colliderect(predator1.hitbox()):
            rewards_predator += 10.0
            rewards_prey -= 10.0
            game_over = True
        else:
            rewards_prey+= 0.1
            if prey1.old_state_x == prey1.new_state_x and prey1.old_state_y == prey1.new_state_y:
                rewards_prey -= 0.2
                rewards_predator += calculate_predator_dense_rewards()


        #Agregar smell, todo este pedazito
        # Check collision with traps
        prey_x = prey1.hitbox().x // constant_variables.tile_size
        prey_y = prey1.hitbox().y // constant_variables.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:
            if world_data[prey_y][prey_x] == 11:
                evasion_probability = prey1.calculate_evasion_probability()
                if random.random() > evasion_probability:
                    print(f"La presa cayó en la trampa. Probabilidad de evadir: {evasion_probability:.2f}")
                    game_over = True
                     # agregar
                    constant_variables.prey_captured = True 
                    data = {
                            **flatten_matrix(state_matrix, "state"),
                            **flatten_matrix(agent_matrix, "agent"),
                            **flatten_matrix(smell_matrix, "smell"),
                            "hunting": predator1.hunting,
                            "prey_x": prey1.new_state_x,
                            "prey_y": prey1.new_state_y,
                            "predator_x": predator1.new_state_x,
                            "predator_y": predator1.new_state_y,
                            "game_over": game_over,
                            "prey_captured": constant_variables.prey_captured if game_over else False,
                            "timeout": constant_variables.timeout if game_over else False,
                        }
                    save_to_csv(data)
                # ---------------------------------------------------------------------------------------------
                else:
                    print(f"La presa EVADIÓ la trampa. Probabilidad de evadir: {evasion_probability:.2f}")
     
            # When predator is over a trap
        pred_x = predator1.hitbox().x // constant_variables.tile_size
        pred_y = predator1.hitbox().y // constant_variables.tile_size
        if 0 <= pred_x < 10 and 0 <= pred_y < 10:
            if world_data[pred_y][pred_x] == 11:
                rewards_predator -= 0.05  # Penalización por pisar trampa
    # Draw agents
    prey1.update()
    predator1.update()
    prey1.draw(window, (255, 255, 0))
    predator1.draw(window, (255, 255, 0))

    if game_over:
        
        font = pygame.font.SysFont(None, 48)
        text = font.render("¡Has perdido! Presiona R", True, (255, 0, 0))
        window.blit(text, (50, constant_variables.height_windows // 2 - 20))
         # agregar
        constant_variables.move_count = 0
        constant_variables.prey_captured = False
        constant_variables.timeout = False
        rewards_prey = 0.0
        rewards_predator = 0.0
        #-----------------------
        
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        
        
        if event.type == pygame.KEYDOWN:
            # PRESA: Solo una dirección a la vez
            if event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                move_up_prey = move_down_prey = move_left_prey = move_right_prey = False
                if event.key == pygame.K_a: move_left_prey = True
                elif event.key == pygame.K_d: move_right_prey = True
                elif event.key == pygame.K_w: move_up_prey = True
                elif event.key == pygame.K_s: move_down_prey = True

            # DEPREDADOR: Solo una dirección a la vez
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                move_up_predator = move_down_predator = move_left_predator = move_right_predator = False
                if event.key == pygame.K_LEFT: move_left_predator = True
                elif event.key == pygame.K_RIGHT: move_right_predator = True
                elif event.key == pygame.K_UP: move_up_predator = True
                elif event.key == pygame.K_DOWN: move_down_predator = True

            # Reinicio del juego
            if event.key == pygame.K_r and game_over:
                world_data, collision_rects, maze, state_matrix, agent_matrix, prey1, predator1, smell_matrix = create_game()
                prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                predator1.predator_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                game_over = False

        
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a: move_left_prey = False
            if event.key == pygame.K_d: move_right_prey = False
            if event.key == pygame.K_w: move_up_prey = False
            if event.key == pygame.K_s: move_down_prey = False
            if event.key == pygame.K_LEFT: move_left_predator = False
            if event.key == pygame.K_RIGHT: move_right_predator = False
            if event.key == pygame.K_UP: move_up_predator = False
            if event.key == pygame.K_DOWN: move_down_predator = False

    pygame.display.update()

pygame.quit()

