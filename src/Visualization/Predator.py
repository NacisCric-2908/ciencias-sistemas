from Agent import Agent
import constant_variables
import math
import pickle
import xgboost as xgb
import os

class Predator(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)
        self.hunting = False
        self.adjusting = False
        self.seen_prey = False
        self.smell_intensity = 0.0
        self.smell_distance = -1
        self.smell_x = -1
        self.smell_y = -1
        self.move = True
        self.obs_predator = [0] * 30
        self.printHumanMode = False


         # Cargar modelo de ML desde ruta absoluta
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "hunting_ML", "hunting_model.pkl")
        #print(f"✅ Charging ML model for Hunting mode: {model_path}")

        with open(model_path, 'rb') as f:
            self.hunting_predictor = pickle.load(f)
        
        #When predator is hunting, the movement apply two states in state matrix. 
        self.hunt1_old_state_x = self.new_state_x
        self.hunt1_old_state_y = self.new_state_y

        self.hunt2_old_state_x = self.new_state_x
        self.hunt2_old_state_y = self.new_state_y


        self.hunt1_new_state_x = self.new_state_x
        self.hunt1_new_state_y = self.new_state_y

        self.hunt2_new_state_x = self.new_state_x
        self.hunt2_new_state_y = self.new_state_y


    def movement(self, delta_x, delta_y ): #Deltas can be (Up, -50) (Down, 50) (Left, 50) (Right, -50)
        if not self.hunting: #Isn´t running
            if self.adjusting and delta_x != 0 and self.shape.x%constant_variables.element_side!=0:
                self.adjust_movement(delta_x, delta_y)

            if self.adjusting and delta_y != 0 and self.shape.x%constant_variables.element_side!=0: 
                if self.printHumanMode: print("Don´t move")
                self.move = False

            if self.adjusting and delta_x != 0 and self.shape.y%constant_variables.element_side!=0:
                if self.printHumanMode: print("Don´t move")
                self.move = False

            if self.adjusting and delta_y != 0 and self.shape.y%constant_variables.element_side!=0: 
                self.adjust_movement(delta_x, delta_y)
                
            if not self.adjusting:
                self.normal_movement(delta_x,delta_y) #Need adjust next move
        else: 
            self.fast_movement(delta_x, delta_y)


    def update_state(self,x,y):
        # Update the state positions
        self.old_state_x = self.new_state_x
        self.old_state_y = self.new_state_y

        if int(self.old_state_x+(x/constant_variables.element_side)) >= 0 and int(self.old_state_x+(x/constant_variables.element_side)) <= 9: 
            self.new_state_x = int(self.old_state_x+(x/constant_variables.element_side))

        if int(self.old_state_y+(y/constant_variables.element_side)) >= 0 and int(self.old_state_y+(y/constant_variables.element_side)) <= 9:
            self.new_state_y = int(self.old_state_y+(y/constant_variables.element_side))


    def update_hunt_state(self,x,y, dir): #Parameter dir, indicates if is in True[x] or False[y]
        self.hunt1_old_state_x = self.hunt1_new_state_x
        self.hunt1_old_state_y = self.hunt1_new_state_y

        self.hunt2_old_state_x = self.hunt2_new_state_x
        self.hunt2_old_state_y = self.hunt2_new_state_y
      
        #Need calculate the new states.
        if x%constant_variables.element_side != 0: #Have two states in x
            pos1_x = x + constant_variables.element_side-(x%constant_variables.element_side) 
            pos2_x = x -(x%constant_variables.element_side)

            self.hunt1_new_state_x = int(pos1_x/constant_variables.element_side)
            self.hunt2_new_state_x = int(pos2_x/constant_variables.element_side)

            self.hunt1_new_state_y = int(y/constant_variables.element_side)
            self.hunt2_new_state_y = int(y/constant_variables.element_side)            

        if y%constant_variables.element_side != 0: #Have two states in y
            pos1_y = y + constant_variables.element_side-(y%constant_variables.element_side) 
            pos2_y = y -(y%constant_variables.element_side)

            self.hunt1_new_state_y = int(pos1_y/constant_variables.element_side)
            self.hunt2_new_state_y = int(pos2_y/constant_variables.element_side)

            self.hunt1_new_state_x = int(x/constant_variables.element_side)
            self.hunt2_new_state_x = int(x/constant_variables.element_side)

        if dir: 
            if x%constant_variables.element_side == 0: #If predator get aline. 
                self.hunt1_new_state_x = int(x/constant_variables.element_side)
                self.hunt2_new_state_x = int(x/constant_variables.element_side)
        else:
            if y%constant_variables.element_side == 0:
                self.hunt1_new_state_y = int(y/constant_variables.element_side)
                self.hunt2_new_state_y = int(y/constant_variables.element_side)


    def update_adjust_state(self,x,y):
        self.new_state_x = int(x/constant_variables.element_side)
        self.new_state_y = int(y/constant_variables.element_side)


    def activate_sensor(self, grid_maze, grid_agents, x, y):
        if not self.hunting and self.adjusting == True: 
            self.fast_sensor(grid_maze, grid_agents)

        elif not self.hunting and self.adjusting == False:
            self.predator_sensor(grid_maze, grid_agents, x, y)

        else:
            self.transition_fast_normal()
            self.fast_sensor(grid_maze, grid_agents)


    def predator_sensor(self, grid_maze, grid_agents, x, y):
        directions = {
            "right": (1, 0),
            "left": (-1, 0),
            "down": (0, 1),
            "up": (0, -1),
            "down_right": (1, 1),
            "down_left": (-1, 1),
            "up_right": (1, -1),
            "up_left": (-1, -1)
        }

        self.seen_prey = False 
        if self.printHumanMode: print(grid_maze[y][x])
        index = 2
        path_index = 0

        if self.printHumanMode: print(self.new_state_x, self.new_state_y)
        self.obs_predator[0] = self.new_state_x
        self.obs_predator[1] = self.new_state_y

        for direction_name, (dx, dy) in directions.items():
            if self.printHumanMode: print(f"\nChecking direction: {direction_name}")
            detected = False

            for paso in range(1, 6):
                nx = x + dx * paso
                ny = y + dy * paso

                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]

                    if valor == 3:
                        if self.printHumanMode: print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                        self.set_obs_predator(3, nx, ny, index)
                        detected = True
                        break

                    elif valor == 5:
                        if grid_agents[ny][nx] == 1:
                            if self.printHumanMode: print(f"Predator: There is a prey at [{nx}, {ny}]")
                            if direction_name in ["up", "down", "left", "right"]:
                                self.seen_prey = True
                                self.transition_normal_fast()
                                self.hunting_direction = direction_name
                            self.set_obs_predator(1, nx, ny, index)
                            detected = True
                            break

                        path_index += 1
                        if path_index == 5:
                            self.set_obs_predator(5, nx, ny, index)
                            detected = True
                            path_index = 0
                            break
                        continue

                    elif valor == 4:
                        if self.printHumanMode: print(f"Predator: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 1:
                            if self.printHumanMode: print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                            if direction_name in ["up", "down", "left", "right"]:
                                self.seen_prey = True
                                self.hunting_direction = direction_name
                            self.set_obs_predator(1, nx, ny, index)
                            detected = True
                            break
                        self.set_obs_predator(4, nx, ny, index)
                        detected = True
                        break

                    else:
                        if self.printHumanMode: print(f"Predator: No vision at [{nx}, {ny}]")

                else:
                    if self.printHumanMode: print(f"Predator: Out of maze at [{nx}, {ny}]")
                    self.set_obs_predator(0, nx, ny, index)
                    detected = True
                    break

            if not detected:
                nx = x + dx
                ny = y + dy
                self.set_obs_predator(0, nx, ny, index)

            index += 3  # Solo una vez por dirección

    def fast_sensor(self, grid_maze, grid_agents):
        directions = {
            "right": (1, 0),
            "left": (-1, 0),
            "down": (0, 1),
            "up": (0, -1),
            "down_right": (1, 1),
            "down_left": (-1, 1),
            "up_right": (1, -1),
            "up_left": (-1, -1)
        }

        x1, y1 = self.hunt1_new_state_x, self.hunt1_new_state_y
        x2, y2 = self.hunt2_new_state_x, self.hunt2_new_state_y

        index = 2  
        self.obs_predator[0] = x1
        self.obs_predator[1] = y1
        self.seen_prey = False

        if x1 == x2 and y1 == y2:
            self._scan_cross_only(x1, y1, grid_maze, grid_agents, directions, index)
        elif y1 == y2:
            self._scan_horizontal_only(x1, x2, y1, grid_maze, grid_agents, directions, index)
        elif x1 == x2:
            self._scan_vertical_only(y1, y2, x1, grid_maze, grid_agents, directions, index)
        else:
            self._fill_no_vision_all(x1, y1, directions, index)

        # Set adjusting flag based on prey detection
        self.adjusting = not self.seen_prey
        if self.seen_prey:
            if self.printHumanMode: print("Sees the prey, can continue hunting")
        else:
            if self.printHumanMode: print("Predator: No prey in sight — switching to normal sensor")

    def _scan_cross_only(self, x, y, grid_maze, grid_agents, directions, index):
        for direction_name in directions:
            dx, dy = directions[direction_name]
            if direction_name in ["up", "down", "left", "right"]:
                if self.printHumanMode: print(direction_name)
                self._scan_direction(direction_name, x, y, dx, dy, grid_maze, grid_agents, index)
            else:
                if self.printHumanMode: print(direction_name)
                self.set_obs_predator(0, x + dx, y + dy, index)  # Sin visión diagonal
            index += 3

    def _scan_horizontal_only(self, x1, x2, y, grid_maze, grid_agents, directions, index):
        left_x = min(x1, x2)
        right_x = max(x1, x2)

        for direction_name in directions:
            dx, dy = directions[direction_name]
            if direction_name == "left":
                if self.printHumanMode: print(direction_name)
                self._scan_direction(direction_name, left_x, y, dx, dy, grid_maze, grid_agents, index)
            elif direction_name == "right":
                if self.printHumanMode: print(direction_name)
                self._scan_direction(direction_name, right_x, y, dx, dy, grid_maze, grid_agents, index)
            else:
                # En horizontal, diagonales y verticales se rellenan con 0
                if self.printHumanMode: print(direction_name)
                self.set_obs_predator(0, left_x + dx, y + dy, index)
            index += 3


    def _scan_vertical_only(self, y1, y2, x, grid_maze, grid_agents, directions, index):
        top_y = min(y1, y2)
        bottom_y = max(y1, y2)

        for direction_name in directions:
            dx, dy = directions[direction_name]
            if direction_name == "up":
                if self.printHumanMode: print(direction_name)
                self._scan_direction(direction_name, x, top_y, dx, dy, grid_maze, grid_agents, index)
            elif direction_name == "down":
                if self.printHumanMode: print(direction_name)
                self._scan_direction(direction_name, x, bottom_y, dx, dy, grid_maze, grid_agents, index)
            else:
                if self.printHumanMode: print(direction_name)
                # En vertical, diagonales y horizontales se rellenan con 0
                self.set_obs_predator(0, x + dx, bottom_y + dy, index)
            index += 3


    def _scan_direction(self, direction_name, x, y, dx, dy, grid_maze, grid_agents, index):
        for paso in range(1, 6):
            nx = x + dx * paso
            ny = y + dy * paso

            if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                valor = grid_maze[ny][nx]
                if valor == 3:
                    if self.printHumanMode: print(f"Wall at [{nx}, {ny}] — vision blocked")
                    self.set_obs_predator(3, nx, ny, index)
                    return
                elif valor == 5:
                    if grid_agents[ny][nx] == 1:
                        if self.printHumanMode: print(f"Prey at [{nx}, {ny}]")
                        if direction_name in ["up", "down", "left", "right"]:
                            self.seen_prey = True
                            self.transition_normal_fast()
                            self.hunting_direction = direction_name
                        self.set_obs_predator(1, nx, ny, index)
                        return
                elif valor == 4:
                    if grid_agents[ny][nx] == 1:
                        if self.printHumanMode: print(f"Prey over trap at [{nx}, {ny}]")
                        if direction_name in ["up", "down", "left", "right"]:
                            self.seen_prey = True
                            self.hunting_direction = direction_name
                        self.set_obs_predator(1, nx, ny, index)
                        return
                    else:
                        self.set_obs_predator(4, nx, ny, index)
                        return
                else:
                    if self.printHumanMode: print(f"No vision at [{nx}, {ny}]")
            else:
                if self.printHumanMode: print(f"Out of maze at [{nx}, {ny}]")
                self.set_obs_predator(0, nx, ny, index)
                return

        # Si no encontró nada en los 5 pasos
        if self.printHumanMode: print(f"Predator: There is path at [{nx}, {ny}]")
        self.set_obs_predator(5, x + dx, y + dy, index)

        
    def normal_movement(self, delta_x, delta_y ):
        # Move in X only if the limits are not exceeded.
        x = self.shape.x + delta_x
        if 0 <= x <= constant_variables.limit_x:
                self.shape.x = x

        # Move in Y only if the limits are not exceeded.
        y = self.shape.y + delta_y
        if 0 <= y <= constant_variables.limit_y:
            self.shape.y = y

        # Update the state positions
        if delta_x != 0 or delta_y != 0:  # Only update state if there is movement
            self.update_state(delta_x, delta_y)


    def fast_movement(self, delta_x, delta_y):
        
        x = self.shape.x + delta_x
        y = self.shape.y + delta_y
        
        if delta_x > delta_y and delta_y == 0: # (Move right)
            if 0 <= x <= constant_variables.limit_x:
                if self.shape.x + delta_x + constant_variables.speed_increase > 450:
                    self.shape.x = 450
                else: 
                    self.shape.x = self.shape.x + delta_x + constant_variables.speed_increase #That 25 is constant
                self.update_hunt_state(self.shape.x, self.shape.y, True)

        if delta_x < delta_y and delta_y == 0: # (Move left)
            if 0 <= x <= constant_variables.limit_x:
                if self.shape.x + delta_x - constant_variables.speed_increase < 0:
                    self.shape.x = 0
                else: 
                    self.shape.x = self.shape.x + delta_x - constant_variables.speed_increase #That 25 is constant
                self.update_hunt_state(self.shape.x, self.shape.y, True)

            
        if delta_y > delta_x and delta_x == 0: # (Move Down)
            if 0 <= y <= constant_variables.limit_y:
                if self.shape.y + delta_y + constant_variables.speed_increase > 450:
                    self.shape.y = 450
                else: 
                    self.shape.y = self.shape.y + delta_y + constant_variables.speed_increase #That 25 is constant
                self.update_hunt_state(self.shape.x, self.shape.y, False)

        if delta_y < delta_x and delta_x == 0: # (Move Up)
            if 0 <= y <= constant_variables.limit_y:
                if self.shape.y + delta_y - constant_variables.speed_increase < 0:
                    self.shape.y = 0
                else: 
                    self.shape.y = self.shape.y + delta_y - constant_variables.speed_increase #That 25 is constant
                self.update_hunt_state(self.shape.x, self.shape.y, False)


    def adjust_movement_collision(self, delta_x, delta_y, distance_move):
        if delta_x>delta_y and delta_y == 0: #Moves in X right
            if self.printHumanMode: print("X = ", self.shape.x, " Y = ", self.shape.y)
            if self.shape.x + constant_variables.element_side - (self.shape.x%constant_variables.element_side) > 450:
                self.shape.x = 450
            else:
                self.shape.x = self.shape.x + distance_move
            self.update_adjust_state(self.shape.x, self.shape.y)

        if delta_x<delta_y and delta_y == 0: #Moves in X left
            if self.shape.x - (self.shape.x%constant_variables.element_side) < 0:
                self.shape.x = 0
            else:
                self.shape.x = self.shape.x - distance_move
            self.update_adjust_state(self.shape.x, self.shape.y)  


        if delta_y>delta_x and delta_x == 0: #Moves in Y down
            if self.shape.y + constant_variables.element_side - (self.shape.y%constant_variables.element_side) > 450:
                self.shape.y = 450
            else:
                self.shape.y = self.shape.y + distance_move
            self.update_adjust_state(self.shape.x, self.shape.y)

                
        if delta_y<delta_x and delta_x == 0: #Moves in Y Up
            if self.shape.y - (self.shape.y%constant_variables.element_side) > 450:
                self.shape.y = 450
            else:
                self.shape.y = self.shape.y - distance_move
            self.update_adjust_state(self.shape.x, self.shape.y)
           
           
    def adjust_movement(self, delta_x, delta_y):
        if delta_x>delta_y and delta_y == 0: #Moves in X right
            if self.shape.x + constant_variables.element_side - (self.shape.x%constant_variables.element_side) > 450:
                self.shape.x = 450
            else:
                self.shape.x = self.shape.x + constant_variables.element_side - (self.shape.x%constant_variables.element_side)
            self.update_adjust_state(self.shape.x, self.shape.y)



        if delta_x<delta_y and delta_y == 0: #Moves in X left
            if self.shape.x - (self.shape.x%constant_variables.element_side) < 0:
                self.shape.x = 0
            else:
                self.shape.x = self.shape.x - (self.shape.x%constant_variables.element_side)
            self.update_adjust_state(self.shape.x, self.shape.y)    


        if delta_y>delta_x and delta_x == 0: #Moves in Y down
            if self.shape.y + constant_variables.element_side - (self.shape.y%constant_variables.element_side) > 450:
                self.shape.y = 450
            else:
                self.shape.y = self.shape.y + constant_variables.element_side - (self.shape.y%constant_variables.element_side)
            self.update_adjust_state(self.shape.x, self.shape.y)

                
        if delta_y<delta_x and delta_x == 0: #Moves in Y Up
            if self.shape.y - (self.shape.y%constant_variables.element_side) > 450:
                self.shape.y = 450
            else:
                self.shape.y = self.shape.y -(self.shape.y%constant_variables.element_side)
            self.update_adjust_state(self.shape.x, self.shape.y)


    def distance(self, a, b):
        if self.printHumanMode: print(abs(a-b))
                 

    def transition_normal_fast(self):
        self.hunt1_new_state_x = self.new_state_x
        self.hunt1_new_state_y = self.new_state_y

        self.hunt2_new_state_x = self.new_state_x
        self.hunt2_new_state_y = self.new_state_y


    def transition_fast_normal(self):
        self.new_state_x = self.hunt1_new_state_x
        self.new_state_y = self.hunt1_new_state_y


    def check_collision_not_zero(self, prev_agent_rect, agent_rect, collision_rects):
        min_distance = 75
        best_collision = None
        best_distance = float('inf')
        best_direction = 'none'

        for rect in collision_rects:
            if agent_rect.colliderect(rect):
                dx = agent_rect.x - prev_agent_rect.x
                dy = agent_rect.y - prev_agent_rect.y

                if dx > 0:  # Derecha
                    direction = 'right'
                    move_amount = dx
                    distance = move_amount + (rect.left - agent_rect.right)
                elif dx < 0:  # Izquierda
                    direction = 'left'
                    move_amount = -dx
                    distance = move_amount + (agent_rect.left - rect.right)
                elif dy > 0:  # Abajo
                    direction = 'down'
                    move_amount = dy
                    distance = move_amount + (rect.top - agent_rect.bottom)
                elif dy < 0:  # Arriba
                    direction = 'up'
                    move_amount = -dy
                    distance = move_amount + (agent_rect.top - rect.bottom)
                else:
                    direction = 'none'
                    distance = 0

                distance = max(0, distance)

                if distance < best_distance:
                    best_distance = distance
                    best_collision = rect
                    best_direction = direction

        if best_collision:
            return True, best_distance, best_direction

        return False, min_distance, "any"


    def check_border_collision(self, prev_agent_rect, agent_rect):
        map_limit = 500  
        min_distance = 75
        best_distance = float('inf')
        best_direction = 'none'

        dx = agent_rect.x - prev_agent_rect.x
        dy = agent_rect.y - prev_agent_rect.y

        if agent_rect.left < 0 and dx < 0:
            direction = 'left'
            distance = prev_agent_rect.left  
            best_distance = distance
            best_direction = direction

        elif agent_rect.right > map_limit and dx > 0:
            direction = 'right'
            distance = map_limit - prev_agent_rect.right
            best_distance = distance
            best_direction = direction

        elif agent_rect.top < 0 and dy < 0:
            direction = 'up'
            distance = prev_agent_rect.top
            best_distance = distance
            best_direction = direction

        elif agent_rect.bottom > map_limit and dy > 0:
            direction = 'down'
            distance = map_limit - prev_agent_rect.bottom
            best_distance = distance
            best_direction = direction

        if best_direction != 'none':
            return True, best_distance, best_direction
        return False, min_distance, "any"



    def smell(self, smell_matrix):
        y0, x0 = self.new_state_y, self.new_state_x 
        rows, cols = 10, 10

        max_intensity = 0.0
        coordinate = None
        distance = 0.0

        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = y0 + dy, x0 + dx

                if 0 <= ny < rows and 0 <= nx < cols:
                    intensity = smell_matrix[ny][nx]
                    if intensity != 0:
                        x_real = nx * 50
                        y_real = ny * 50
                        x0_real = x0 * 50
                        y0_real = y0 * 50

                        normal_distance = math.sqrt((x_real - x0_real) ** 2 + (y_real - y0_real) ** 2)

                        value = 300 - (normal_distance / intensity)

                        if value > max_intensity:
                            max_intensity = value
                            coordinate = (nx, ny)
                            distance = normal_distance

        self.smell_intensity = max_intensity

        self.obs_predator[26] = 0 if not self.hunting else 1

        if coordinate is not None:
            self.set_obs_predator(max_intensity, coordinate[0], coordinate[1], 27)
            if self.printHumanMode: print(f"Max Intensity: {max_intensity:.2f}, Coordinates: {coordinate}, Distance: {distance:.2f}")
            self.smell_distance = distance
            self.smell_x = coordinate[0]
            self.smell_y = coordinate[1]
        else:
            # Si no hay olor, pon un marcador neutral
            self.smell_distance = -1
            self.smell_x = -1
            self.smell_y = -1   
            self.set_obs_predator(0, -1, -1, 27)
            if self.printHumanMode: print("Not smell nothing.")

        if self.printHumanMode: print(self.obs_predator)

        
    def should_hunt(self):
        if self.printHumanMode: print(self.smell_intensity)
        if self.printHumanMode: print(self.seen_prey)
        input_data = [[self.smell_intensity, self.seen_prey]]
        self.hunting = self.hunting_predictor.predict(input_data)[0] == 1
        if self.printHumanMode: print("Hunting mode: ", self.hunting)


    def set_obs_predator(self, obj, x, y, index):
        if self.printHumanMode: print(f"Setting observation at index {index}: Object={obj}, X={x}, Y={y}")

        self.obs_predator[index] = obj
        self.obs_predator[index + 1] = x
        self.obs_predator[index + 2] = y 

    def _fill_no_vision_all(self, x, y, directions, index):
        for direction_name in directions:
            dx, dy = directions[direction_name]
            self.set_obs_predator(0, x + dx, y + dy, index)
            index += 3