from Agent import Agent
import constant_variables
import math
import pickle
import xgboost as xgb


class Predator(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)
        self.hunting = False
        self.adjusting = False
        self.seen_prey = False
        self.smell_intensity = 0.0
        self.move = True

        #ML model will take the decision of hunting or not.
        self.hunting_predictor = None
        with open('src/Visualization/hunting_ML/hunting_model.pkl', 'rb') as f:
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
                print("Don´t move")
                self.move = False

            if self.adjusting and delta_x != 0 and self.shape.y%constant_variables.element_side!=0:
                print("Don´t move")
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

        opposite_direction = {
            "right": "left",
            "left": "right",
            "up": "down",
            "down": "up"
        }

        if not self.hunting:
            self.seen_prey = False 
            print(grid_maze[y][x])

            for direction_name, (dx, dy) in directions.items():
                print(f"\nChecking direction: {direction_name}")
                for paso in range(1, 6):
                    nx = x + dx * paso
                    ny = y + dy * paso

                    if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                        valor = grid_maze[ny][nx]
                        if valor == 3:
                            print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                            break
                        elif valor == 5:
                            if grid_agents[ny][nx] == 1:
                                print(f"Predator: There is a prey at [{nx}, {ny}]")
                                if direction_name in ["up", "down", "left", "right"]:
                                    self.seen_prey = True
                                    self.transition_normal_fast()
                                    self.hunting_direction = direction_name
                                break
                            continue
                        elif valor == 4:
                            print(f"Predator: There is a trap at [{nx}, {ny}]")
                            if grid_agents[ny][nx] == 1:
                                print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                                if direction_name in ["up", "down", "left", "right"]:
                                    self.seen_prey = True
                                    self.hunting_direction = direction_name
                                break
                            continue
                        else:
                            print(f"Predator: No vision at [{nx}, {ny}]")
                    else:
                        print(f"Predator: Out of maze at [{nx}, {ny}]")
                        break

        else:
            print("Predator is hunting, focused vision mode")

            # Obtener dirección y su opuesta
            hunting_dirs = [self.hunting_direction, opposite_direction[self.hunting_direction]]
            found_prey = False

            for direction_name in hunting_dirs:
                dx, dy = directions[direction_name]
                print(f"\nFocused scan: {direction_name}")

                for paso in range(1, 6):
                    nx = x + dx * paso
                    ny = y + dy * paso

                    if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                        valor = grid_maze[ny][nx]
                        if valor == 3:
                            print(f"Blocked by wall at [{nx}, {ny}]")


    def fast_sensor(self, grid_maze, grid_agents):
        detect_prey = False
        #Possible positions of predator in state matrix
        x1, y1 = self.hunt1_new_state_x, self.hunt1_new_state_y
        x2, y2 = self.hunt2_new_state_x, self.hunt2_new_state_y

        #If predator is in mode hunting and one cell
        if x1 == x2 and y1 == y2:
            directions = {
                "right": (1, 0),
                "left": (-1, 0),
                "down": (0, 1),
                "up": (0, -1)
            }
            for dir_name, (dx, dy) in directions.items():
                print(f"\nChecking direction: {dir_name}")
                for paso in range(1, 6):
                    nx = x1 + dx * paso
                    ny = y1 + dy * paso

                    if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                        valor = grid_maze[ny][nx]
                        if valor == 3:
                            print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")                
                            break
                        elif valor == 5:
                            if grid_agents[ny][nx] == 1:
                                print(f"Predator: There is a prey at [{nx}, {ny}]")
                                detect_prey = True
                            continue
                        elif valor == 4:
                            print(f"Predator: There is a trap at [{nx}, {ny}]")
                            if grid_agents[ny][nx] == 1:
                                print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                                detect_prey = True
                            continue
                        else:
                            print(f"Predator: No vision at [{nx}, {ny}]")
                    else:
                        print(f"Predator: Out of maze at [{nx}, {ny}]")
                        break

            if not detect_prey:
                self.seen_prey = False
                print("Predator: No prey in sight — switching to normal sensor")
                return
            else:
                self.seen_prey = True
                print("Continue mode hunting")
                return

        # Determine if is detecting horizontal or vertical
        if y1 == y2:
            # Direction horizontal
            left_x = min(x1, x2)
            right_x = max(x1, x2)
            y = y1

            print(f"\nChecking direction: left")
            for paso in range(1, 6):
                nx = left_x - paso
                ny = y
                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]
                    if valor == 3:
                        print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                        break
                    elif valor == 5:
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a prey at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    elif valor == 4:
                        print(f"Predator: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                            detect_prey = True
                        continue
                    else:
                        print(f"Predator: No vision at [{nx}, {ny}]")
                else:
                    print(f"Predator: Out of maze at [{nx}, {ny}]")
                    break

            # Ver hacia la derecha desde la casilla más a la derecha
            print(f"\nChecking direction: right")
            for paso in range(1, 6):
                nx = right_x + paso
                ny = y
                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]
                    if valor == 3:
                        print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                        break
                    elif valor == 5:
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a prey at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    elif valor == 4:
                        print(f"Predator: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    else:
                        print(f"Predator: No vision at [{nx}, {ny}]")
                else:
                    print(f"Predator: Out of maze at [{nx}, {ny}]")
                    break

        elif x1 == x2:
            # Dirección vertical
            top_y = min(y1, y2)
            bottom_y = max(y1, y2)
            x = x1

            # Ver hacia arriba desde la casilla superior
            print(f"\nChecking direction: up")
            for paso in range(1, 6):
                nx = x
                ny = top_y - paso
                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]
                    if valor == 3:
                        print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                        break
                    elif valor == 5:
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a prey at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    elif valor == 4:
                        print(f"Predator: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    else:
                        print(f"Predator: No vision at [{nx}, {ny}]")
                else:
                    print(f"Predator: Out of maze at [{nx}, {ny}]")
                    break

            # Ver hacia abajo desde la casilla inferior
            print(f"\nChecking direction: down")
            for paso in range(1, 6):
                nx = x
                ny = bottom_y + paso
                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]
                    if valor == 3:
                        print(f"Predator: There is a wall at [{nx}, {ny}] — vision blocked")
                        break
                    elif valor == 5:
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a prey at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    elif valor == 4:
                        print(f"Predator: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 1:
                            print(f"Predator: There is a Prey over trap at [{nx}, {ny}]")
                            detect_prey = True
                            break
                        continue
                    else:
                        print(f"Predator: No vision at [{nx}, {ny}]")
                else:
                    print(f"Predator: Out of maze at [{nx}, {ny}]")
                    break

        if not detect_prey:
                self.seen_prey = False
                self.adjusting = True
                print("Predator: No prey in sight — switching to normal sensor")
                return
        else:
                self.seen_prey = True
                self.adjusting = False
                print("Sees the prey, can continue hunting")
                return

    
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
            print("X = ", self.shape.x, " Y = ", self.shape.y)
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
        print(abs(a-b))
                 

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

        if coordinate is not None:
            print(f"Max Intensity: {max_intensity:.2f}, Coordinates: {coordinate}, Distance: {distance:.2f}")
        else:
            print("Not smell nothing.")

        self.smell_intensity = max_intensity
        return coordinate, max_intensity, distance

    def should_hunt(self):
        print(self.smell_intensity)
        print(self.seen_prey)
        input_data = [[self.smell_intensity, self.seen_prey]]
        self.hunting = self.hunting_predictor.predict(input_data)[0] == 1
        print("Hunting mode: ", self.hunting)