from Agent import Agent
import constant_variables

class Predator(Agent):
    def __init__(self, x, y, animation):
        super().__init__(x, y, animation)
        
    def movement(self, delta_x, delta_y ): #Deltas can be (Up, -50) (Down, 50) (Left, 50) (Right, -50)
        # Move in X only if the limits are not exceeded.
        x = self.shape.x + delta_x
        if 0 <= x <= constant_variables.limit_x:
            self.shape.x = x

        # Move in Y only if the limits are not exceeded.
        y = self.shape.y + delta_y
        if 0 <= y <= constant_variables.limit_y:
            self.shape.y = y

    def update_state(self,x,y):
        print("Hello from Predator")