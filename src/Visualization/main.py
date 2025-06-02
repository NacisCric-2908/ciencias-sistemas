import pygame
import constant_variables
from Agent import Agent
from Maze import Maze
from Prey import Prey
from Predator import Predator
import random

# Start the game
pygame.init()

# Show main window (width, height)
window = pygame.display.set_mode((constant_variables.width_windows, constant_variables.height_windows))

# Put a name
pygame.display.set_caption("Prey Predator")

# Prey animation
animation_prey = []
for i in range(7):
    img_prey = pygame.image.load(f"src/Visualization/assets/images/prey/Prey_{i}.png")
    img_prey = pygame.transform.scale(img_prey, (50, 50))
    animation_prey.append(img_prey)

# Predator animation
animation_predator = []
for i in range(7):
    img_predator = pygame.image.load(f"src/Visualization/assets/images/predator/Predator_{i}.png")
    img_predator = pygame.transform.scale(img_predator, (50, 50))
    animation_predator.append(img_predator)

# Tile images
tile_list = []
for x in range(12):
    tile_image = pygame.image.load(f"src/Visualization/assets/images/tiles/Tile ({x+1}).png")
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
        print("No se pudo colocar todas las trampas respetando las restricciones.")

def create_game():
    world_data = [row.copy() for row in initial_world_data]
    randomize_traps(world_data)
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
    update_state_matrix_from_world(world_data, state_matrix)
    state_matrix[9][0] = 1  # Prey
    state_matrix[0][9] = 2  # Predator
    prey1 = Prey(25, 475, animation_prey)
    predator1 = Predator(475, 25, animation_predator)
    return world_data, collision_rects, maze, state_matrix, prey1, predator1

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

def print_state_matrix(state_matrix):
    print("Matriz de estado:")
    for row in state_matrix:
        print(row)
    print("-" * 40)

def draw_grid():
    for x in range(10):
        pygame.draw.line(window, (203, 50, 52), (x * constant_variables.tile_size, 0),
                         (x * constant_variables.tile_size, constant_variables.height_windows))
        pygame.draw.line(window, (203, 50, 52), (0, x * constant_variables.tile_size),
                         (constant_variables.width_windows, x * constant_variables.tile_size))

def check_collision(agent_rect, collision_rects):
    for rect in collision_rects:
        if agent_rect.colliderect(rect):
            return True
    return False

# Initialize game
world_data, collision_rects, maze, state_matrix, prey1, predator1 = create_game()
prey1.prey_sensor(state_matrix, prey1.new_state_x, prey1.new_state_y)

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

while run:
    clock.tick(constant_variables.FPS)
    window.fill(constant_variables.color_back)
    draw_grid()

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
        if not check_collision(prey1.hitbox().move(delta_x_prey, delta_y_prey), collision_rects):
            prey1.movement(delta_x_prey, delta_y_prey)
            if delta_x_prey != 0 or delta_y_prey != 0:
                update_state_matrix_from_world(world_data, state_matrix)
                prey1.prey_sensor(state_matrix, prey1.new_state_x, prey1.new_state_y)
                print_state_matrix(state_matrix)

        # Move predator
        if not check_collision(predator1.hitbox().move(delta_x_predator, delta_y_predator), collision_rects):
            predator1.movement(delta_x_predator, delta_y_predator)

        # Check collision with predator
        if prey1.hitbox().colliderect(predator1.hitbox()):
            game_over = True

        # Check collision with traps
        prey_x = prey1.hitbox().x // constant_variables.tile_size
        prey_y = prey1.hitbox().y // constant_variables.tile_size
        if 0 <= prey_x < 10 and 0 <= prey_y < 10:
            if world_data[prey_y][prey_x] == 11:
                game_over = True

    # Draw agents
    prey1.update()
    predator1.update()
    prey1.draw(window, (255, 255, 0))
    predator1.draw(window, (255, 255, 0))

    if game_over:
        font = pygame.font.SysFont(None, 48)
        text = font.render("¡Has perdido! Presiona R", True, (255, 0, 0))
        window.blit(text, (50, constant_variables.height_windows // 2 - 20))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a: move_left_prey = True
            if event.key == pygame.K_d: move_right_prey = True
            if event.key == pygame.K_w: move_up_prey = True
            if event.key == pygame.K_s: move_down_prey = True
            if event.key == pygame.K_LEFT: move_left_predator = True
            if event.key == pygame.K_RIGHT: move_right_predator = True
            if event.key == pygame.K_UP: move_up_predator = True
            if event.key == pygame.K_DOWN: move_down_predator = True
            if event.key == pygame.K_r and game_over:
                world_data, collision_rects, maze, state_matrix, prey1, predator1 = create_game()
                prey1.prey_sensor(state_matrix, prey1.new_state_x, prey1.new_state_y)
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
