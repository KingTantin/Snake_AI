import game as game 
import numpy as np
from ppo_torch import Agent
import matplotlib.pyplot as plt
import time

    

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
    learning_rate = 0.0003
    value_coef = 0.8
    entropy_coef = 0.05
    policy_clip = 0.2
    num_games = 100000


    #Initializing variables
    score_history = []
    best_avg_score = 0
    best_score = 0
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    current_learning_step = 0
    perfect_games = 0

    #for plotting
    episode_list = []
    avg_score_list = []


    env = game.snake_environment(width, height, move_limit, block_size, visual=False)



    n_actions = env.num_actions()
    
    

    input_dims = env.observation_space_shape()
    

    agent = Agent(n_actions = n_actions, batch_size = batch_size, learning_rate = learning_rate, num_epochs = num_learning_epochs, input_dims = input_dims, policy_clip=policy_clip, value_coef=value_coef, entropy_coef=entropy_coef)
    #agent.load_models('file_path')
    agent.load_models() #-> loads the files in netoworks




    for episode in range(num_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:

            action, prob, val = agent.choose_action(observation)
            

            observations_, reward, done = env.step(action)


            n_steps += 1
            score += reward

            

            agent.remember(observation, action, prob, val, reward, done)


            if n_steps % N == 0:

                agent.learn()
                learn_iters += 1
                episode_list.append(learn_iters)
                avg_score_list.append(avg_score)

            if episode % 1000 == 0 and done == False:
                env.draw_game(observations_)
                
                

            observation = observations_
        
        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])



        if reward > 0:
            perfect_games += 1
        
        if score > best_score:
            best_score = score

        if avg_score > best_avg_score and episode > 100:
            best_avg_score = avg_score

            if current_learning_step < learn_iters:
                current_learning_step = learn_iters
                agent.save_models()


        if perfect_games > 0:
            print(f'episode: {episode} \tscore: {round(score, 2)} \t avg_score: {round(avg_score, 2)} \t, best_avg_score: {round(best_avg_score, 2)} \t learning_steps: {learn_iters} \t perfect_games: {perfect_games}')
        else:
            print(f'episode: {episode} \tscore: {round(score, 2)} \t avg_score: {round(avg_score, 2)} \t, best_avg_score: {round(best_avg_score, 2)} \t learning_steps: {learn_iters} \t best_score: {best_score}')

    plt.plot(episode_list, avg_score_list)
    plt.savefig('Snake_AI/Score_Graph.png')