import pygame
import constant_variables
from Agent import Agent

#Start the game
pygame.init() 

#Show main window (width, height)
window = pygame.display.set_mode((constant_variables.width_windows,constant_variables.height_windows))

#Put a the name
pygame.display.set_caption("Prey Depredator")

#Is a array of prey images
animation_prey = []
for i in range (7):
    #Insert an image, searching relative path
    img_prey = pygame.image.load(f"src/Visualization/assets/images/prey/Prey_{i}.png" )
    img_prey = pygame.transform.scale(img_prey, (50,50)) #Change image size
    animation_prey.append(img_prey) #Save all images

#Is a array of predator images
animation_predator = []
for i in range (7):
    #Insert an image, searching relative path
    img_predator = pygame.image.load(f"src/Visualization/assets/images/predator/Predator_{i}.png" )
    img_predator = pygame.transform.scale(img_predator, (50,50)) #Change image size
    animation_predator.append(img_predator) #Save all images

#Creating the agent, receive initial (x,y) and the array images
prey1 = Agent(25,25, animation_prey)
predator1 = Agent(475,25, animation_predator)

#variables moves prey
move_up_prey = False
move_down_prey = False
move_left_prey = False
move_right_prey = False

#variables moves predator
move_up_predator = False
move_down_predator = False
move_left_predator = False
move_right_predator = False

#Clock for control the frame rate
clock = pygame.time.Clock()

run = True 

while run: 

    #Run at n FPS
    clock.tick(constant_variables.FPS)


    #Clean the movements
    window.fill(constant_variables.color_back)

    delta_x_prey = 0 #How many move in x
    delta_y_prey = 0 #How many move in y

    delta_x_predator = 0 #How many move in x
    delta_y_predator = 0 #How many move in y

    #Calculate the movement of PREY
    if move_right_prey == True:
        delta_x_prey = constant_variables.speed_prey
    if move_left_prey == True:
        delta_x_prey = -constant_variables.speed_prey
    if move_up_prey == True:
        delta_y_prey = -constant_variables.speed_prey #Take care with coordinates in pygame -up
    if move_down_prey == True:
        delta_y_prey = constant_variables.speed_prey #Take care with coordinates in pygame +down

    #Calculate the movement of PREDATOR
    if move_right_predator == True:
        delta_x_predator = constant_variables.speed_predator
    if move_left_predator == True:
        delta_x_predator = -constant_variables.speed_predator
    if move_up_predator == True:
        delta_y_predator = -constant_variables.speed_predator #Take care with coordinates in pygame -up
    if move_down_predator == True:
        delta_y_predator = constant_variables.speed_predator #Take care with coordinates in pygame +down

    #print(f"{delta_x_prey},{delta_y_prey}")

    #Move the prey, parameters coordinate now
    prey1.movement(delta_x_prey,delta_y_prey)

    #Move the predator, parameters coordinate now
    predator1.movement(delta_x_predator,delta_y_predator)

    #Update the image
    prey1.update() 

    #Update the image
    predator1.update() 

    prey1.draw(window, (255,255,0)) #Draw the prey in window
   
    predator1.draw(window, (255,255,0)) #Draw the prey in window
    
    for event in pygame.event.get(): #Always run the game
        if event.type == pygame.QUIT: #Finish when close 
            run = False

        #Here is the order keyboard for movement PREY
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a: 
                move_left_prey = True
            if event.key == pygame.K_d: 
                move_right_prey = True
            if event.key == pygame.K_w: 
                move_up_prey = True
            if event.key == pygame.K_s: 
                move_down_prey = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a: 
                move_left_prey = False
            if event.key == pygame.K_d: 
                move_right_prey = False
            if event.key == pygame.K_w: 
                move_up_prey = False
            if event.key == pygame.K_s: 
                move_down_prey = False


        #Here is the order keyboard for movement PREDATOR
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT: 
                move_left_predator = True
            if event.key == pygame.K_RIGHT: 
                move_right_predator = True
            if event.key == pygame.K_UP: 
                move_up_predator = True
            if event.key == pygame.K_DOWN: 
                move_down_predator = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT: 
                move_left_predator = False
            if event.key == pygame.K_RIGHT: 
                move_right_predator = False
            if event.key == pygame.K_UP: 
                move_up_predator = False
            if event.key == pygame.K_DOWN: 
                move_down_predator = False


    pygame.display.update() #IMPORTANT, update the window to show constant changes

pygame.quit #Finish the visualization 