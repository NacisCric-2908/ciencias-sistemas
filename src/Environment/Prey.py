from Environment.Agent import Agent
import Environment.constant_variables as constant_variables
import math
import pygame

class Prey(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)  # Call the parent constructor
        self.start_time = pygame.time.get_ticks()
        self.seen_predator = False
        self.fatigue = 0
        self.obs_prey = [0] * 30
        self.printHumanMode = False

   #Change the coordinates for movement
    def movement(self, delta_x, delta_y ): #Deltas can be (Up, -50) (Down, 50) (Left, 50) (Right, -50)

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


    def update_state(self,x,y):
        # Update the state positions
        self.old_state_x = self.new_state_x
        self.old_state_y = self.new_state_y

        if int(self.old_state_x+(x/50)) >= 0 and int(self.old_state_x+(x/50)) <= 9: 
            self.new_state_x = int(self.old_state_x+(x/50))

        if int(self.old_state_y+(y/50)) >= 0 and int(self.old_state_y+(y/50)) <= 9:
            self.new_state_y = int(self.old_state_y+(y/50))

        # Function that activates the prey sensor
    def prey_sensor(self, grid_maze, grid_agents, x, y):
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

        self.seen_predator = False
        if self.printHumanMode: print(grid_maze[y][x])  # Debug print
        self.obs_prey = [0] * 30  # Reinicia siempre el vector
        self.obs_prey[0] = self.new_state_x
        self.obs_prey[1] = self.new_state_y

        index = 2
        path_index = 0

        for direction_name, (dx, dy) in directions.items():
            if self.printHumanMode: print(f"\nChecking direction: {direction_name}")
            detected = False  # Controla si ya se asignó algo en esta dirección

            for paso in range(1, 6):
                nx = x + dx * paso
                ny = y + dy * paso

                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]

                    if valor == 3:  # Wall
                        if self.printHumanMode: print(f"Prey: There is a wall at [{nx}, {ny}] — vision blocked")
                        self.set_obs_prey(3, nx, ny, index)
                        detected = True
                        break

                    elif valor == 5:  # Path
                        if grid_agents[ny][nx] == 2:
                            self.seen_predator = True
                            if self.printHumanMode: print(f"Prey: There is a predator at [{nx}, {ny}]")
                            self.set_obs_prey(2, nx, ny, index)
                            detected = True
                            break

                        path_index += 1
                        if path_index == 5:
                            self.set_obs_prey(5, nx, ny, index)
                            detected = True
                            path_index = 0
                        continue

                    elif valor == 4:  # Trap
                        if self.printHumanMode: print(f"Prey: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 2:
                            if self.printHumanMode: print(f"Prey: There is a Predator over trap at [{nx}, {ny}]")
                            self.seen_predator = True
                            self.set_obs_prey(2, nx, ny, index)
                            detected = True
                            break
                        self.set_obs_prey(4, nx, ny, index)
                        detected = True
                        break

                    else:
                        if self.printHumanMode: print(f"Prey: No vision at [{nx}, {ny}]")

                else:
                    if self.printHumanMode: print(f"Prey: Out of maze at [{nx}, {ny}]")
                    self.set_obs_prey(0, nx, ny, index)
                    detected = True
                    break

            # If doesn't see anything
            if not detected:
                nx = x + dx
                ny = y + dy
                self.set_obs_prey(0, nx, ny, index)

            index += 3  

        if self.fatigue == 3:
            self.seen_predator = False

        if self.printHumanMode: print(self.obs_prey)

    def calculate_evasion_probability(self):
            elapsed_ms = pygame.time.get_ticks() - self.start_time
            t = elapsed_ms / 1000  # Convert to seconds

            P_base = constant_variables.prey_evasion_base
            B = constant_variables.prey_learning_rate

            probability = P_base * (1 - math.exp(-B * t))
            return probability
    
    def set_obs_prey(self, obj, x, y, index):
        if self.printHumanMode: print(f"Setting observation at index {index}: Object={obj}, X={x}, Y={y}")

        self.obs_prey[index] = obj
        self.obs_prey[index + 1] = x
        self.obs_prey[index + 2] = y 
