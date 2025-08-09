# utils.py
import Environment.constant_variables as constant_variables
import pygame
import random
import os
import pygame

# Get path to the current directory
BASE_PATH = os.path.dirname(__file__)

# 1. Charge sprites for prey
def load_prey_sprites():
    animation_prey = []
    for i in range(7):
        img = pygame.image.load(f"src/assets/images/prey/Prey_{i}.png")
        img = pygame.transform.scale(img, (50, 50))
        animation_prey.append(img)
    return animation_prey

# 2. Charge sprites for predator
def load_predator_sprites():
    animation_predator = []
    for i in range(7):
        img = pygame.image.load(f"src/assets/images/predator/Predator_{i}.png")
        img = pygame.transform.scale(img, (50, 50))
        animation_predator.append(img)
    return animation_predator

# 3. Charge tiles for the maze
def load_tiles():
    tile_list = []
    for x in range(12):
        img = pygame.image.load(f"src/assets/images/tiles/Tile ({x+1}).png")
        img = pygame.transform.scale(img, (constant_variables.tile_size, constant_variables.tile_size))
        tile_list.append(img)
    return tile_list

# 4. Original world matrix (before randomizing traps)
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

# 5. Randomize traps in the world matrix
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

# 6. Function to assign random and distant initial positions
def generate_valid_start_positions(world_data, tile_size=50, min_distance=6):
    valid_positions = []
    for y, row in enumerate(world_data):
        for x, tile in enumerate(row):
            if tile in [8, 9, 10]:  # Valid paths
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
                return {
                    "prey_px": prey_px, "prey_py": prey_py,
                    "prey_x_init": prey_pos[0], "prey_y_init": prey_pos[1],
                    "predator_px": predator_px, "predator_py": predator_py,
                    "predator_x_init": predator_pos[0], "predator_y_init": predator_pos[1]
                }
    raise ValueError("No se encontraron posiciones iniciales válidas tras múltiples intentos.")
