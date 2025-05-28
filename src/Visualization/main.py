import pygame
import constant_variables
from Agent import Agent
from Maze import Maze
from Prey import Prey
from Predator import Predator

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

#Charge maze images
tile_list = []
for x in range(12):
    tile_image = pygame.image.load(f"src/Visualization/assets/images/tiles/Tile ({x+1}).png")
    tile_image = pygame.transform.scale(tile_image, (constant_variables.tile_size, constant_variables.tile_size)) #Change image size
    tile_list.append(tile_image) #Save all images

#Is a matrix of the maze/world
world_data = [
    [11,7,8,9,10,9,10,8,10,9],
    [9,4,4,11,7,10,4,4,4,8],
    [8,4,4,9,6,9,5,11,5,10],
    [10,6,8,9,10,9,5,8,5,9],
    [8,9,10,0,4,9,6,8,6,10],
    [10,7,8,9,4,9,10,8,10,9],
    [8,5,11,9,2,3,3,2,1,8],
    [9,5,8,9,10,9,10,8,10,9],
    [10,6,10,0,2,3,1,9,0,3],
    [10,9,8,9,10,9,10,8,10,11]
]

#Create the walls collisions
collision_rects = []

for y, row in enumerate(world_data):
    for x, tile in enumerate(row):
        if tile in constant_variables.collision_tiles: #If the tile is a collision tile
            rect = pygame.Rect(x * constant_variables.tile_size, y * constant_variables.tile_size, constant_variables.tile_size, constant_variables.tile_size)
            collision_rects.append(rect)


#This function evaluate the collision of the agent with the walls
def check_collision(agent_rect):
    for rect in collision_rects:
        if agent_rect.colliderect(rect):
            return True
    return False


#Create the maze
maze = Maze()
maze.process_data(world_data, tile_list) #Process the data of the maze


#Creating the maze
def draw_grid():
    for x in range(10):
        pygame.draw.line(window, (203,50,52), (x*constant_variables.tile_size, 0), (x*constant_variables.tile_size, constant_variables.height_windows))
        pygame.draw.line(window, (203,50,52), (0, x*constant_variables.tile_size), (constant_variables.width_windows, x*constant_variables.tile_size))


'''
Have a states matrix
0 - No vision
1 - Prey
2 - Predator
3 - Wall
4 - Trap
5 - Path
'''

state_matrix = [
    [4,3,5,5,5,5,5,2,2,5],
    [5,3,3,4,5,5,3,3,3,5],
    [5,3,3,5,3,5,3,4,3,5],
    [5,3,5,5,5,5,3,5,3,5],
    [5,5,5,3,3,5,3,5,3,5],
    [5,3,5,5,3,5,5,5,5,5],
    [5,3,4,5,3,3,3,3,3,5],
    [5,3,5,5,5,5,5,5,5,5],
    [5,3,5,3,3,3,3,5,3,3],
    [1,5,5,5,5,5,5,5,5,4]
]

def update_state_matrix(state_matrix, prey_x, prey_y, new_x, new_y):
 
    x = prey_x #Current state x
    y = prey_y #Current state y
    new_x = new_x #New state x
    new_y = new_y #New state y

    #Change states
    aux_value = state_matrix[y][x] 

    state_matrix[y][x] = state_matrix[new_y][new_x]   

    state_matrix[new_y][new_x] = aux_value 

    print("Matrix updated")


#Creating the agent, receive initial (x,y) and the array images
prey1 = Prey(25,475, animation_prey)
predator1 = Predator(475,25, animation_predator)

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

####First activation of the prey sensor###########################################
update_state_matrix(state_matrix, prey1.old_state_x , prey1.old_state_y ,prey1.new_state_x, prey1.new_state_y)

for row in state_matrix:
    print(row)

prey1.prey_sensor(state_matrix, prey1.new_state_x, prey1.new_state_y) #Activate the prey sensor

print("Prey state matrix:", prey1.old_state_x, prey1.old_state_y, "->", prey1.new_state_x, prey1.new_state_y)

######################################################################################


#Clock for control the frame rate
clock = pygame.time.Clock()

run = True 

while run: 

    #Run at n FPS
    clock.tick(constant_variables.FPS)

    #Clean the movements
    window.fill(constant_variables.color_back)

    draw_grid() #Draw the grid

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

    #Draw the maze in window
    maze.draw(window) 

    #Move the prey, parameters coordinate now
    #print(f"Prey: {prey1.hitbox().move(delta_x_prey,delta_y_prey)}") #To show the steps
    if not check_collision(prey1.hitbox().move(delta_x_prey,delta_y_prey)):
        prey1.movement(delta_x_prey,delta_y_prey)

        if delta_x_prey != 0 or delta_y_prey != 0: #If the prey is moving
            update_state_matrix(state_matrix, prey1.old_state_x , prey1.old_state_y ,prey1.new_state_x, prey1.new_state_y)
            for row in state_matrix:
                print(row)

            prey1.prey_sensor(state_matrix, prey1.new_state_x, prey1.new_state_y) #Activate the prey sensor
        

    #Move the predator, parameters coordinate now
    if not check_collision(predator1.hitbox().move(delta_x_predator,delta_y_predator)):
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