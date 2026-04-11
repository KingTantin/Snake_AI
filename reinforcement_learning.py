import game as game 
import numpy as np
from ppo_torch import Agent
import matplotlib.pyplot as plt
import time
#changes since last commit: -add more Hyperparameters to reinforcement_learning -changed file paths to Snake_Ai  - fix drawing function  - better generation of apples
    

if __name__ == '__main__':
   
    
    #Hyperparameters Snake
    width = 6
    height = 6
    move_limit = 36
    block_size = 100
    
    #Hyperparameters AI
    N = 1024
    batch_size = 128
    num_learning_epochs = 3
    learning_rate = 0.0002
    value_coef = 0.7
    entropy_coef = 0.07
    policy_clip = 0.3
    num_games = 100000


    #Initializing variables
    score_history = []
    best_avg_score = 0
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    current_learning_step = 0

    #for plotting
    episode_list = []
    avg_score_list = []


    env = game.snake_environment(width, height, move_limit, block_size, visual=False)



    n_actions = env.num_actions()
    
    

    input_dims = env.observation_space_shape()
    

    agent = Agent(n_actions = n_actions, batch_size = batch_size, learning_rate = learning_rate, num_epochs = num_learning_epochs, input_dims = input_dims, policy_clip=policy_clip, value_coef=value_coef, entropy_coef=entropy_coef)
    #agent.load_models('/home/constantin/Downloads/Coding_Projects/Artifical_Intelligence/Snake_AI/best_6x6_networks')
    agent.load_models() #-> loads the files in netoworks




    for episode in range(num_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:

            action, prob, val = agent.choose_action(observation)
            

            observations_, reward, done = env.step(action)
            


            n_steps += 1
            score += np.mean(reward)

            

            agent.remember(observation, action, prob, val, reward, done)


            if n_steps % N == 0:

                agent.learn()
                learn_iters += 1
                episode_list.append(learn_iters)
                avg_score_list.append(avg_score)

            if episode % 2000000 == 0 and done == False:
                env.draw_game()
                
                

            observation = observations_
        
        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])




        if avg_score > best_avg_score and episode > 100:
            best_avg_score = avg_score

            if current_learning_step < learn_iters:
                current_learning_step = learn_iters
                agent.save_models()

        print(f'episode: {episode} \tscore: {round(score, 2)} \t avg_score: {round(avg_score, 2)} \t, best_avg_score: {round(best_avg_score, 2)} \t time_steps: {n_steps} \t learning_steps: {learn_iters}')

    plt.plot(episode_list, avg_score_list)
    plt.savefig('Snake_AI/Score_Graph.png')