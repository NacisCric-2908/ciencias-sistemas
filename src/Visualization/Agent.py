import pygame
import constant_variables

#The class Agent has the main visual characteristics of agents. 
class Agent(): 
    def __init__(self, x, y, animation):
        self.animation = animation #Array of images
        self.frame_index = 0 #Iterator of images array
        self.update_time = pygame.time.get_ticks() #Time now
        self.image = animation[self.frame_index] #Image now
        self.shape = pygame.Rect(0,0,constant_variables.height_element,constant_variables.width_element)
        self.shape.center = (x,y)

    def update(self):
        #The time to change the image
        cooldown_animation = 150

        #Calculate the time passed to change the image
        if pygame.time.get_ticks() - self.update_time >= cooldown_animation: 
            self.frame_index = self.frame_index + 1
            self.update_time = pygame.time.get_ticks()

        #Restart the iterator
        if self.frame_index >= len(self.animation):
            self.frame_index = 0
        
        #Update the image
        self.image = self.animation[self.frame_index]

    #Change the coordinates for movement
    def movement(self, delta_x, delta_y ):
        # Move in X only if the limits are not exceeded.
        x = self.shape.x + delta_x
        if 0 <= x <= constant_variables.limit_x:
            self.shape.x = x

        # Move in Y only if the limits are not exceeded.
        y = self.shape.y + delta_y
        if 0 <= y <= constant_variables.limit_y:
            self.shape.y = y

    #Show in the window
    def draw(self, window, color):
        window.blit(self.image, self.shape)
        #pygame.draw.rect(window, color, self.shape, 1)