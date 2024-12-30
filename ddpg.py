import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, max_action): 
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300) 
        self.fc3 = nn.Linear(300, action_dim) 
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x)) 
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

env = gym.make('Humanoid-v4')
env_human = gym.make('Humanoid-v4', render_mode='human')

# Get dimensions of state and action spaces
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize networks
actor = Actor(state_dim, action_dim, max_action)
actor_target = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)

# Copy the weights to target networks
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

# Initialize replay buffer
replay_buffer = ReplayBuffer(max_size=1000000)

def ddpg_update(batch_size, gamma=0.99, tau=0.005):
    # Sample a batch of transitions
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to PyTorch tensors
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # Compute target Q-value
    with torch.no_grad():
        next_actions = actor_target(next_states)
        target_Q = critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * gamma * target_Q

    # Optimize the critic
    current_Q = critic(states, actions)
    critic_loss = F.mse_loss(current_Q, target_Q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Optimize the actor
    actor_loss = -critic(states, actor(states)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Update the target networks
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


num_episodes = 1000
batch_size = 64
rewards = []

for episode in range(num_episodes):
    episode_reward = 0
    if episode - 990 >= 0:
        state, _ = env_human.reset()
    else:
        state, _ = env.reset()

    while True:

        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)  
        action = actor(state_tensor).detach().numpy()[0]
        
        # Add gaussian noise 
        noise = np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action) if episode - 990 < 0 else env_human.step(action)
        done = terminated or truncated
        replay_buffer.add((state, action, reward, next_state, done))
        
        if len(replay_buffer.buffer) > batch_size:
            ddpg_update(batch_size)

        state = next_state
        episode_reward += reward

        if done:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")
    rewards.append(episode_reward)
    

env.close()

mean_rewards = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
plt.plot(mean_rewards)
plt.xlabel('Episode')
plt.ylabel('Mean Reward (last 100 episodes)')
plt.title('Training Progress')
plt.show()

