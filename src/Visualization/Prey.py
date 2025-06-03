from Agent import Agent
import constant_variables

class Prey(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)  # Call the parent constructor

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
    def prey_sensor(self, grid_maze, grid_agents, x, y):  # Parameters are x and y coordinates of the prey
        # This sensor checks the surroundings of the prey
        directions = {  # Direction dictionary
            "right": (1, 0),
            "left": (-1, 0),
            "down": (0, 1),
            "up": (0, -1),
            "down_right": (1, 1),
            "down_left": (-1, 1),
            "up_right": (1, -1),
            "up_left": (-1, -1)
        }

        print(grid_maze[y][x])  # Debug print of current cell (optional)
        

        for direction_name, (dx, dy) in directions.items():
            print(f"\nChecking direction: {direction_name}")
            for paso in range(1, 6):
                nx = x + dx * paso  # New column
                ny = y + dy * paso  # New row

                # Verify if the new coordinates are inside the maze
                if 0 <= ny < len(grid_maze) and 0 <= nx < len(grid_maze[0]):
                    valor = grid_maze[ny][nx]
                    if valor == 3:  # Wall
                        print(f"Prey: There is a wall at [{nx}, {ny}] — vision blocked")
                        break
                    elif valor == 5:  # Path
                        if grid_agents[ny][nx] == 2:  # There is a predator
                            print(f"Prey: There is a predator at [{nx}, {ny}]")
                            break                           
                        continue  # path is clear
                    elif valor == 4:  # Trap
                        print(f"Prey: There is a trap at [{nx}, {ny}]")
                        if grid_agents[ny][nx] == 2:  # There is a predator
                            print(f"Prey: There is a Predator over trap at [{nx}, {ny}]")
                            break
                        continue
                    else:
                        print(f"Prey: No vision at [{nx}, {ny}]")
                else:
                    print(f"Prey: Out of maze at [{nx}, {ny}]")
                    break
