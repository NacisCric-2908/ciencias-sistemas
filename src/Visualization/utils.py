# utils.py

import pygame
import constant_variables
import random

# 1. Función para cargar sprites de la presa
def load_prey_sprites():
    animation_prey = []
    for i in range(7):
        img = pygame.image.load(f"src/Visualization/assets/images/prey/Prey_{i}.png")
        img = pygame.transform.scale(img, (50, 50))
        animation_prey.append(img)
    return animation_prey

# 2. Función para cargar sprites del depredador
def load_predator_sprites():
    animation_predator = []
    for i in range(7):
        img = pygame.image.load(f"src/Visualization/assets/images/predator/Predator_{i}.png")
        img = pygame.transform.scale(img, (50, 50))
        animation_predator.append(img)
    return animation_predator

# 3. Función para cargar las tiles del entorno
def load_tiles():
    tile_list = []
    for x in range(12):
        img = pygame.image.load(f"src/Visualization/assets/images/tiles/Tile ({x+1}).png")
        img = pygame.transform.scale(img, (constant_variables.tile_size, constant_variables.tile_size))
        tile_list.append(img)
    return tile_list

# 4. Matriz del mundo original (antes de randomizar trampas)
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

# 5. Función para randomizar las trampas en el mapa
def randomize_traps(world_data):
    trap_positions = []
    path_positions = []
    for y, row in enumerate(world_data):
        for x, tile in enumerate(row):
            if tile == 11:
                trap_positions.append((x, y))
            elif tile in [8, 9, 10]:
                path_positions.append((x, y))

    prey_start = (0, 9)
    predator_start = (9, 0)

    new_trap_positions = []
    max_attempts = 1000
    attempts = 0
    while len(new_trap_positions) < len(trap_positions) and attempts < max_attempts:
        attempts += 1
        random.shuffle(path_positions)
        valid_position = path_positions.pop()
        too_close = False
        for pos in new_trap_positions:
            if abs(pos[0] - valid_position[0]) <= 1 and abs(pos[1] - valid_position[1]) <= 1:
                too_close = True
                break
        if too_close or valid_position in [prey_start, predator_start]:
            continue
        new_trap_positions.append(valid_position)

    if len(new_trap_positions) == len(trap_positions):
        for y, row in enumerate(world_data):
            for x, tile in enumerate(row):
                if tile == 11:
                    world_data[y][x] = random.choice([8, 9, 10])
        for x, y in new_trap_positions:
            world_data[y][x] = 11
    else:
        print("Can't put the traps correctly.")
