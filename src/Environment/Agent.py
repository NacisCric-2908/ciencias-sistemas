import pygame
import Environment.constant_variables as constant_variables

#The class Agent has the main visual characteristics of agents. 

class Agent(): 
    def __init__(self, x, y, animation):
        self.animation = animation #Array of images
        self.frame_index = 0 #Iterator of images array
        self.update_time = pygame.time.get_ticks() #Time now
        if animation:
            self.image = animation[self.frame_index]
        else:
            self.image = None  # o una imagen por defecto si existe
        self.shape = pygame.Rect(0,0,constant_variables.height_element,constant_variables.width_element)
        self.shape.center = (x,y)

        self.old_state_x = int((self.shape.x/constant_variables.size_prey)) 
        self.old_state_y = int((self.shape.y/constant_variables.size_prey)) 
        self.new_state_x = int((self.shape.x/constant_variables.size_prey)) 
        self.new_state_y = int((self.shape.y/constant_variables.size_prey))

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

    def movement(self, delta_x, delta_y ):
        raise NotImplementedError("Move like a prey or predator")
    
    def update_state(self,x,y):
        raise NotImplementedError("Update state of prey or predator")

    #Show in the window
    def draw(self, window, color):
        window.blit(self.image, self.shape)
        pygame.draw.rect(window, color, self.shape, 1)

    def hitbox(self):
        return self.shape
