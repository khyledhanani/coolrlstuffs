import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, actions_dim):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, 64)
        self.hidden_layer = nn.Linear(64, 128)
        self.output_layer = nn.Linear(128, actions_dim) 

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        action_logits = self.output_layer(x) 
        return action_logits

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        value = self.output_layer(x)
        return value

class ActorCritic:
    def __init__(self, obs_dim, action_dim):
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = ValueNetwork(obs_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=0.0001)
        self.gamma = 0.98

    def sample_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze()

    def update(self, state, action, reward, next_state, done, log_prob):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([float(done)])

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state).detach()

        # Compute TD error
        if done:
            delta_t = reward
        else:
            delta_t = reward + self.gamma * next_value

        td_error = delta_t - value

        # Critic loss
        critic_loss = F.mse_loss(value, td_error.detach().mean())
        
        # Actor loss
        actor_loss = -(log_prob * td_error.detach()).mean()


        # Backprop 
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        
        critic_loss.backward()
        actor_loss.backward()
        
        self.actor_opt.step()
        self.critic_opt.step()

        return critic_loss.item(), actor_loss.item()

lunar_lander_params = {'continuous': False, 'gravity': -10.0, 'enable_wind': False, 'wind_power': 1.0, 'turbulence_power':1}
env = gym.make( "LunarLander-v2", **lunar_lander_params)
env_human = gym.make( "LunarLander-v2", render_mode="human", **lunar_lander_params)
title="LunarLander-v2"

num_episodes = 1000
render_last_n_episodes = 10
agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
rewards = []

if __name__ == "__main__":
    for episode in range(num_episodes):
        if episode >= num_episodes - render_last_n_episodes:
            state, _ = env_human.reset()
        else:
            state, _ = env.reset()
            
        episode_reward = 0
        critic_losses = []
        actor_losses = []

        for step in range(1000):
            action, log_prob = agent.sample_action(state)
            if episode >= num_episodes - render_last_n_episodes:
                next_state, reward, terminated, truncated, _ = env_human.step(action)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
                
            done = terminated or truncated

            critic_loss, actor_loss = agent.update(state, action, reward, next_state, done, log_prob)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Critic Loss: {np.mean(critic_losses):.4f}, "
                  f"Actor Loss: {np.mean(actor_losses):.4f}")

    env.close()
    env_human.close()

    mean_rewards = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
    plt.plot(mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (last 100 episodes)')
    plt.title('Training Progress')
    plt.show()