#Window variables
width_windows = 500
height_windows = 500

#Prey Variables
prey_x = 375
prey_y = 375
prey_x_init = int((prey_x-25)/50)
prey_y_init = int((prey_y-25)/50)

speed_prey = 50
size_prey = 50 
smell_evaporation_interval = 5  # Cuántos frames deben pasar antes de evaporar el olor
smell_initial_strength = 5      # Valor de olor que deja la presa
smell_previous_strength = 4     # Valor para la celda anterior
smell_evaporation_counter = 0
prey_evasion_base = 0.7  # P_base
prey_learning_rate = 0.2  # B
prey_captured = False
moveout = False
timeout = False

#Predator Variables
predator_x = 125
predator_y = 25
predator_x_init = int((predator_x-25)/50)
predator_y_init = int((predator_y-25)/50)

speed_predator = 50
size_predator = 50 
speed_increase = 25
predator_moved = False

#Environment Variables
tile_size = 50 #Size of the tile
collision_tiles = [0, 1, 2, 3, 4, 5, 6, 7] #Tiles that are collisions of walls
max_moves = 45  # Límite de movimientos por partida
move_count = 0    # Contador global

#Elements Variables
element_side = 50
width_element = 50
height_element = 50
limit_y = height_windows-height_element 
limit_x = width_windows-width_element
color_back = (0,0,20) #Color of window
FPS = 10 #Frames per second
