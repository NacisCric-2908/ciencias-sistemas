import pygame
import constant_variables
from Agent import Agent
from Maze import Maze
from Prey import Prey
from Predator import Predator
import numpy as np
import random
import csv
import os

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
    agent_matrix = np.zeros_like(state_matrix)
    smell_matrix = [[0 for _ in range(10)] for _ in range(10)]

    update_state_matrix_from_world(world_data, state_matrix)
    agent_matrix[9][0] = 1  # Prey
    agent_matrix[0][9] = 2  # Predator
    prey1 = Prey(25, 475, animation_prey)
    predator1 = Predator(475, 25, animation_predator)
    return world_data, collision_rects, maze, state_matrix, agent_matrix , prey1, predator1, smell_matrix

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

def update_agent_matrix_fast(agent_matrix): #Update the matrix when Predator is in mode Hunting
    agent_matrix[predator1.new_state_y][predator1.new_state_x] = 0 
    agent_matrix[predator1.hunt1_old_state_y][predator1.hunt1_old_state_x] = 0
    agent_matrix[predator1.hunt2_old_state_y][predator1.hunt2_old_state_x] = 0
    agent_matrix[predator1.hunt1_new_state_y][predator1.hunt1_new_state_x] = 2
    agent_matrix[predator1.hunt2_new_state_y][predator1.hunt2_new_state_x] = 2

def update_agent_matrix_adjust(agent_matrix): #Update the matrix when Predator need adjusting
    agent_matrix[predator1.hunt1_new_state_y][predator1.hunt1_new_state_x] = 0
    agent_matrix[predator1.hunt2_new_state_y][predator1.hunt2_new_state_x] = 0
    agent_matrix[predator1.new_state_y][predator1.new_state_x] = 2
    predator1.adjusting = False

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
    for rect in collision_rects:
        if agent_rect.colliderect(rect):
            return True
    return False

# Initialize game
world_data, collision_rects, maze, state_matrix, agent_matrix, prey1, predator1, smell_matrix = create_game()
prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)

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
        if not check_collision(prey1.hitbox().move(delta_x_prey, delta_y_prey), collision_rects): #Not move in collisions
            prey1.movement(delta_x_prey, delta_y_prey)
            smell_matrix[prey1.new_state_y][prey1.new_state_x] = constant_variables.smell_initial_strength
           
            if delta_x_prey != 0 or delta_y_prey != 0:
                 # agregar
                constant_variables.move_count += 1
                if constant_variables.move_count >= constant_variables.max_moves:
                    game_over = True
                    constant_variables.moveout = True
                    constant_variables.prey_captured = False

                if prey1.hitbox().colliderect(predator1.hitbox()):
                    game_over = True
                    constant_variables.prey_captured = True
                    constant_variables.timeout = False
    
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
                "timeout": constant_variables.moveout if game_over else False,             
                }
                save_to_csv(data)
            #--------------------------------------------    
                update_agent_matrix(agent_matrix, prey1.old_state_x, prey1.old_state_y, prey1.new_state_x, prey1.new_state_y)
                prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                predator1.smell(smell_matrix)
                predator1.should_hunt()  # Check if predator should hunt based on ML model
                print_state_matrix(state_matrix)
                print_agent_matrix(agent_matrix)
                print_smell_matrix(smell_matrix)
                
        # Move predator 
        collision_x, collision_y = delta_x_predator, delta_y_predator 

        if delta_x_predator != 0 or delta_y_predator != 0: #Only moves (50,0)(-50,0)(0,-50)(0,50)
            # agregar
            constant_variables.move_count += 1
            
            if constant_variables.move_count >= constant_variables.max_moves:
                game_over = True
                constant_variables.moveout = True
                constant_variables.prey_captured = False
                
            if prey1.hitbox().colliderect(predator1.hitbox()):
                game_over = True
                constant_variables.prey_captured = True
                constant_variables.timeout = False
                
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
                "prey_captured": constant_variables.prey_captured if game_over else False,  # Solo relevante si game_over=True
                "timeout": constant_variables.moveout if game_over else False,             # Solo relevante si game_over=True
                }
            save_to_csv(data)
            # ---------------------------------------------------------------
            if predator1.hunting: #If hunting the move is + 25, moves (75,0)(-75,0)(0,-75)(0,75)
                if delta_x_predator > delta_y_predator and delta_y_predator == 0: #Move right
                    collision_x = delta_x_predator + constant_variables.speed_increase
                if delta_x_predator < delta_y_predator and delta_y_predator == 0: #Move left
                    collision_x = delta_x_predator-constant_variables.speed_increase
                if delta_y_predator > delta_x_predator and delta_x_predator == 0: #Move down
                    collision_y = delta_y_predator+constant_variables.speed_increase
                if delta_y_predator < delta_x_predator and delta_x_predator == 0: #Move up
                    collision_y = delta_y_predator-constant_variables.speed_increase

            #Check if collision with walls
            collision, distance, direction = predator1.check_collision_not_zero(predator1.hitbox(), predator1.hitbox().move(collision_x, collision_y), collision_rects)
            
            #Check if collision with maze borders
            collision_border, distance_border, direction_border = predator1.check_border_collision(predator1.hitbox(), predator1.hitbox().move(collision_x, collision_y))
            
            if not collision_border:
                if not collision: 
                    
                    predator1.movement(delta_x_predator, delta_y_predator)
                    
                    if predator1.move: #Special case, not move descentralized. 
                        if predator1.hunting: 
                            update_agent_matrix_fast(agent_matrix)
                        else:
                            if predator1.adjusting:
                                update_agent_matrix_adjust(agent_matrix)
                            else: 
                                predator1.transition_normal_fast()
                                update_agent_matrix(agent_matrix, predator1.old_state_x, predator1.old_state_y, predator1.new_state_x, predator1.new_state_y)

                        prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                        predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                        predator1.smell(smell_matrix)
                        predator1.should_hunt()  # Check if predator should hunt based on ML model
                        print_state_matrix(state_matrix)
                        print_agent_matrix(agent_matrix)
                        print_smell_matrix(smell_matrix)
                        
                    
                    else:
                        print("[Don´t move]")
                        predator1.move = True
                    
                else:
                    if distance == 0:
                        print("[Collision]?", collision)
                        print_state_matrix(state_matrix)
                        print_agent_matrix(agent_matrix)
                        print_smell_matrix(smell_matrix)


                    elif distance == 25:
                        predator1.adjust_movement_collision(collision_x, collision_y, distance)
                        update_agent_matrix_adjust(agent_matrix)
                        predator1.transition_normal_fast()
                        prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                        predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                        predator1.smell(smell_matrix)
                        predator1.should_hunt()  # Check if predator should hunt based on ML model
                        print_state_matrix(state_matrix)
                        print_agent_matrix(agent_matrix)
                        print_smell_matrix(smell_matrix)

                    elif distance == 50:
                        
                        if predator1.adjusting and collision_y != 0 and predator1.shape.x%50!=0: 
                            print("Don´t move")
                            predator1.move = False

                        if predator1.adjusting and collision_x != 0 and predator1.shape.y%50!=0:
                            print("Don´t move")
                            predator1.move = False

                        if predator1.adjusting and collision_x != 0 and predator1.shape.x%50!=0:
                            predator1.adjust_movement_collision(collision_x, collision_y, distance)
                            update_agent_matrix_adjust(agent_matrix)
                            predator1.transition_normal_fast()
                            prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                            predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                            predator1.smell(smell_matrix)
                            predator1.should_hunt()  # Check if predator should hunt based on ML model
                            print_state_matrix(state_matrix)
                            print_agent_matrix(agent_matrix)
                            print_smell_matrix(smell_matrix)
                            
                        if predator1.adjusting and collision_y != 0 and predator1.shape.y%50!=0: 
                            predator1.adjust_movement_collision(collision_x, collision_y, distance)
                            update_agent_matrix_adjust(agent_matrix)
                            predator1.transition_normal_fast()
                            prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                            predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                            predator1.smell(smell_matrix)
                            predator1.should_hunt()  # Check if predator should hunt based on ML model
                            print_state_matrix(state_matrix)
                            print_agent_matrix(agent_matrix)
                            print_smell_matrix(smell_matrix)

                        if not predator1.adjusting and predator1.shape.y%50==0 and predator1.shape.x%50==0 :
                            predator1.adjust_movement_collision(collision_x, collision_y, distance)
                            update_agent_matrix_adjust(agent_matrix)
                            predator1.transition_normal_fast()
                            prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                            predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                            predator1.smell(smell_matrix)
                            predator1.should_hunt()  # Check if predator should hunt based on ML model
                            print_state_matrix(state_matrix)
                            print_agent_matrix(agent_matrix)
                            print_smell_matrix(smell_matrix)

                    else:
                        print("Debug")

            else: 
                if distance_border == 0:
                    print("[Collision border]?", collision_border)
                    print_state_matrix(state_matrix)
                    print_agent_matrix(agent_matrix)
                    print_smell_matrix(smell_matrix)

                elif distance_border == 25:
                    predator1.adjust_movement_collision(collision_x, collision_y, distance_border)
                    update_agent_matrix_adjust(agent_matrix)
                    predator1.transition_normal_fast()
                    prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                    predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                    predator1.smell(smell_matrix)   
                    predator1.should_hunt()  # Check if predator should hunt based on ML model
                    
                    print_state_matrix(state_matrix)
                    print_agent_matrix(agent_matrix)
                    print_smell_matrix(smell_matrix)

                elif distance_border == 50:

                    predator1.adjust_movement_collision(collision_x, collision_y, distance_border)
                    update_agent_matrix_adjust(agent_matrix)
                    predator1.transition_normal_fast()
                    prey1.prey_sensor(state_matrix, agent_matrix, prey1.new_state_x, prey1.new_state_y)
                    predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
                    predator1.smell(smell_matrix)
                    predator1.should_hunt()  # Check if predator should hunt based on ML model
                    print_state_matrix(state_matrix)
                    print_agent_matrix(agent_matrix)
                    print_smell_matrix(smell_matrix)
                else:
                    print("Debug")
                    
        
        # Check collision with predator
        if prey1.hitbox().colliderect(predator1.hitbox()):
            game_over = True


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
                predator1.activate_sensor(state_matrix, agent_matrix, predator1.new_state_x, predator1.new_state_y)
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