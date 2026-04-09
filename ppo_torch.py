import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        #batch_start = np.array([i for i in range(0, n_states, self.batch_size)])
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(0, n_states)#np.array([i for i in range(0, n_states)])
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, prob , value, action, reward, done):
        self.states.append(state)
        self.probs.append(prob)
        self.values.append(value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []


class PolicyNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, fc1_dims = 256, fc2_dims = 256, checkpoint_dir = 'Snake/networks'):
        super(PolicyNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, 'policy_torch_ppo')
        print(self.checkpoint_file)
        self.policy = nn.Sequential(nn.Linear(input_dims, fc1_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc1_dims, fc2_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc2_dims, fc2_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc2_dims, n_actions),
                                    nn.Softmax(dim=-1))
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.policy(state)
        dist = Categorical(dist)

        return dist
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

        
    def load_checkpoint(self, new_checkpoint_dir = ''):
        if new_checkpoint_dir:
            checkpoint_file = os.path.join(new_checkpoint_dir, 'policy_torch_ppo')
            self.load_state_dict(torch.load(checkpoint_file))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims = 256, fc2_dims = 256, checkpoint_dir = 'Snake/networks'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, 'critic_torch_ppo')
        
        self.critic = nn.Sequential(nn.Linear(input_dims, fc1_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc1_dims, fc2_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc2_dims, fc2_dims),
                                    nn.ReLU(),
                                    nn.Linear(fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)



    def forward(self, state):
        value = self.critic(state)

        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

        
    def load_checkpoint(self, new_checkpoint_dir = ''):
        if new_checkpoint_dir:
            checkpoint_file = os.path.join(new_checkpoint_dir, 'critic_torch_ppo')
            self.load_state_dict(torch.load(checkpoint_file))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma = 0.99, learning_rate = 0.0003, gae_lambda = 0.95,  policy_clip = 0.2, batch_size = 64, num_epochs = 10, value_coef = 0.7, entropy_coef = 0.05):#0.6 0.03
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = PolicyNetwork(n_actions, input_dims, learning_rate)
        self.critic = CriticNetwork(input_dims, learning_rate)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, prob, value, reward, done):
        self.memory.store_memory(state, prob, value, action, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, checkpoint_dir = ''):
        print('... loading models ...')
        if checkpoint_dir:
            self.policy.load_checkpoint(checkpoint_dir)
            self.critic.load_checkpoint(checkpoint_dir)
        else:
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()

    def choose_action(self, observation):


        # Convert to tensor once, add batch dimension, keep on correct device
        state = torch.tensor(observation, dtype=torch.float, device=self.policy.device).unsqueeze(0)

        # Forward pass through policy & critic
        dist = self.policy(state)
        value = self.critic(state)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        #Convert back to Python scalars (single step)
        return action.item(), log_prob.item(), value.item()
        


    def learn(self):

        state_arr, action_arr, old_probs_arr, values_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

        N = len(reward_arr)
        reward_tensor = torch.tensor(reward_arr, device=self.policy.device)
        done_tensor = torch.tensor(done_arr, dtype=torch.float32, device=self.policy.device)
        values_torch = torch.tensor(values_arr, device=self.policy.device)

        
        
        #Compute advantage
        advantage = torch.zeros(N, device=self.policy.device)
        td_errors = reward_tensor[:-1] + self.gamma * values_torch[1:] * (1 - done_tensor[:-1]) - values_torch[:-1]

        gae = 0
        for t in reversed(range(N-1)):
            gae = td_errors[t] + self.gamma * self.gae_lambda * (1 - done_tensor[t]) * gae
            advantage[t] = gae

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        

        states_torch = torch.tensor(state_arr, dtype = torch.float).to(self.policy.device)
        old_probs_torch = torch.tensor(old_probs_arr).to(self.policy.device)
        actions_torch = torch.tensor(action_arr).to(self.policy.device)


        for _ in range(self.num_epochs):

            #self.policy.optimizer.zero_grad()
            #self.critic.optimizer.zero_grad()
            for batch in batches:
                
                states = states_torch[batch]
                old_probs = old_probs_torch[batch]
                actions = actions_torch[batch]


                dist = self.policy(states)
                
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)  #maybe not needed


                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs-old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                policy_loss = -1*torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values_torch[batch]
                critic_loss = torch.square(returns - critic_value).mean()

                

                entropy = dist.entropy()
    
                total_loss = policy_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

                self.policy.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                self.policy.optimizer.step()
                self.critic.optimizer.step()
            state_arr, action_arr, old_probs_arr, values_arr, reward_arr, done_arr, batches = self.memory.generate_batches()



        self.memory.clear_memory()