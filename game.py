import pygame
import numpy as np
import random
class snake_environement():
    def __init__(self, width, height, move_limit, block_size, visual = False):
        self.width = width
        self.height = height


        

        self.move_limit = move_limit
        self.block_size = block_size
        self.visual = visual
        self.state_layout = [0 for i in range(self.width*self.height+1)]  
        
        


        pixel_width = width * block_size
        pixel_height = height * block_size
        
        if self.visual:
            self.screen = pygame.display.set_mode((pixel_width, pixel_height))


    def num_actions(self):
        return 3


    def observation_space_shape(self):
        
        return self.height * self.width + 1
        
    

    def get_game_vector(self):

        state = self.state_layout.copy()       
        max_length = self.width*self.height

        
        
        for pos_index in range(len(self.snake_body)-1):

            state[self.width*self.snake_body[pos_index][1]+self.snake_body[pos_index][0]] = (max_length-pos_index)/(max_length+1) #pos_index+3

        

        state[self.width*self.snake_body[-1][1]+self.snake_body[-1][0]] = 2
        state[self.width*self.apple_position[1]+self.apple_position[0]] = 1
        state[-1] = 1-(self.move_counter/self.move_limit)

        return state
        
        





    def random_apple_position(self, new_head_position = []):
        while True:
            apple_position = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
            if apple_position not in self.snake_body and apple_position != new_head_position:
                break
        return apple_position

    

    def reset(self):

        self.move_counter = 0

        
        self.snake_body = [[self.width//8, self.height//2], [self.width//8+1, self.height//2]]



        self.apple_position = self.random_apple_position()



        self.current_direction = [1, 0]

        self.length = 2


        state = self.get_game_vector()

        
        return state


    def get_relative_direction(self, action: int):
        if action == 0:
            #keep current direction
            return self.current_direction
        
        
        elif action == 1:
            #turn right
            return [-self.current_direction[1], self.current_direction[0]]
            
        elif action == 2:
            #turn left
            return [self.current_direction[1], -self.current_direction[0]]


    def reward_function(self):

        distance_before_action = np.sqrt((self.apple_position[0]-self.snake_body[-2][0])**2+(self.apple_position[1]-self.snake_body[-2][1])**2)
        distance_after_action = np.sqrt((self.apple_position[0]-self.snake_body[-1][0])**2+(self.apple_position[1]-self.snake_body[-1][1])**2)

        return (distance_before_action-distance_after_action)*0.01

    def step(self, action: int):


        done = False
        self.move_counter += 1

        new_head_position = [0, 0]
        old_head_position = [self.snake_body[-1][0], self.snake_body[-1][1]]

        self.current_direction = self.get_relative_direction(action) 

        new_head_position[0] = self.current_direction[0] + old_head_position[0]
        new_head_position[1] = self.current_direction[1] + old_head_position[1]


        

   
        


        #Terminal state
        if new_head_position[0] > (self.width-1) or new_head_position[1] > (self.height-1) or new_head_position[0] < 0 or new_head_position[1] < 0 or new_head_position in self.snake_body:

            reward = -10
            done = True


        elif self.move_counter >  self.move_limit:

            reward = -10
            done = True


        elif new_head_position == self.apple_position:
            reward = 10 + self.length - 0.015*self.move_counter
            self.apple_position = self.random_apple_position(new_head_position=new_head_position)
            self.length += 1
            self.move_counter = 0


        else:
            reward = self.reward_function()

        
        #Update Snake Body
        self.snake_body.append(new_head_position)       
        if self.length< len(self.snake_body):
            self.snake_body = self.snake_body[1: len(self.snake_body)]
            #self.batch_snakes[self.game_index].pop(0)


        if done:
            state = [0 for i in range(self.width*self.height+1)]
        else:
            state = self.get_game_vector()


       


        
        return state, reward, done

    def draw_game(self):
        if self.visual:
            input_vector = self.get_game_vector()
            clock = pygame.time.Clock()

            for num in range(len(input_vector)):
                pos_width = (num % self.width)*self.block_size
                pos_height = (num // (self.width)) * self.block_size


                if input_vector[num] == 1:
                    color = [255, 0, 0]
                elif input_vector[num] == 2:
                    color = [255, 0, 255]
                elif input_vector[num] > 2:

                    #color1 = 200-(160//self.length)*(input_vector[num]-2)
                    # color3 = 120-(80//self.length)*(input_vector[num]-2)
                    color1 = 150
                    color3 = 150
                    color = [color1, 0, color3]
                else:
                    color = [0, 0, 0]
                pygame.draw.rect(self.screen, color, (pos_width, pos_height, self.block_size, self.block_size))
            pygame.display.update()
            clock.tick(15)
        else:
            print('You need to set visual True') 