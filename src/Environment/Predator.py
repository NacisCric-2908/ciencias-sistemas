from Environment.Agent import Agent
import Environment.constant_variables as constant_variables
import math
import pickle
import xgboost as xgb
import os

class Predator(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)
        self.hunting = False
        self.seen_prey = False
        self.smell_intensity = 0.0
        self.smell_distance = -1
        self.smell_x = -1
        self.smell_y = -1
        self.move = True
        self.obs_predator = [0] * 30
        self.printHumanMode = False


        #Charge the ML model for Hunting mode
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CURRENT_DIR, '..', 'hunting_ML', 'hunting_model.pkl')
        #print(f"✅ Charging ML model for Hunting mode: {model_path}")

        with open(model_path, 'rb') as f:
            self.hunting_predictor = pickle.load(f)
        
    def movement(self, delta_x, delta_y): #Deltas can be (Up, -50) (Down, 50) (Left, 50) (Right, -50)
         #Isn´t running
        self.normal_movement(delta_x,delta_y) #Self.normal_movement((delta_x), (delta_y))eed adjust next move 


    def update_state(self,x,y):
        # Update the state positions
        self.old_state_x = self.new_state_x
        self.old_state_y = self.new_state_y

        if int(self.old_state_x+(x/constant_variables.element_side)) >= 0 and int(self.old_state_x+(x/constant_variables.element_side)) <= 9: 
            self.new_state_x = int(self.old_state_x+(x/constant_variables.element_side))

        if int(self.old_state_y+(y/constant_variables.element_side)) >= 0 and int(self.old_state_y+(y/constant_variables.element_side)) <= 9:
            self.new_state_y = int(self.old_state_y+(y/constant_variables.element_side))

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

            index += 3  
        
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
            #If there is no smell, set default values
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