import pygame
import constant_variables

#The class Agent has the main visual characteristics of agents. 
class Agent(): 
    def __init__(self, x, y):
        self.shape = pygame.Rect(0,0,constant_variables.height_element,constant_variables.width_element)
        self.shape.center = (x,y)

    #Change the coordinates for movement
    def movement(self, delta_x, delta_y ):
        self.shape.x = self.shape.x + delta_x
        self.shape.y = self.shape.y + delta_y

    #Show in the window
    def draw(self, window, color):
        pygame.draw.rect(window, color, self.shape)