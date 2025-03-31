import pygame
import constant_variables
from Agent import Agent

#Start the game
pygame.init() 

#Show main window (width, height)
window = pygame.display.set_mode((constant_variables.width_windows,constant_variables.height_windows))

#Put a the name
pygame.display.set_caption("Prey Depredator")

#Is a array of images
animation = []
for i in range (7):
    #Insert an image, searching relative path
    img = pygame.image.load(f"src/Visualization/assets/images/prey/Prey_{i}.png" )
    img = pygame.transform.scale(img, (50,50)) #Change image size
    animation.append(img) #Save all images

#Creating the agent, receive initial (x,y) and the array images
prey1 = Agent(25,25, animation)
#prey2 = Agent(90,30)

#variables moves prey
move_up_prey = False
move_down_prey = False
move_left_prey = False
move_right_prey = False

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

    #Calculate the movement of PREY
    if move_right_prey == True:
        delta_x_prey = constant_variables.speed_prey
    if move_left_prey == True:
        delta_x_prey = -constant_variables.speed_prey
    if move_up_prey == True:
        delta_y_prey = -constant_variables.speed_prey #Take care with coordinates in pygame -up
    if move_down_prey == True:
        delta_y_prey = constant_variables.speed_prey #Take care with coordinates in pygame +down

    print(f"{delta_x_prey},{delta_y_prey}")

    #Move the prey, parameters coordinate now
    prey1.movement(delta_x_prey,delta_y_prey)

    #Update the image
    prey1.update() 

    prey1.draw(window, (255,255,0)) #Draw the prey in window
    #prey2.draw(window, (255,255,255)) #Draw the prey in window
   
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

    pygame.display.update() #IMPORTANT, update the window to show constant changes

pygame.quit #Finish the visualization 