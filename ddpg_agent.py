import numpy as np
import random
import copy
from collections import namedtuple, deque
from OUNoise import OUNoise

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size   1e5
BATCH_SIZE = 128        # minibatch size       128
GAMMA = 0.99            # discount factor      0.99
TAU = 2e-3              # for soft update of target parameters   1e-3 and 2e-3 work
LR_ACTOR = 1e-3         # learning rate of the actor   1e-3 
LR_CRITIC = 1e-4        # learning rate of the critic  2e-4
WEIGHT_DECAY = 0        # L2 weight decay              0
LEARN_EVERY = 4         #4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Multiple agent DDPG, interacts with and learns from the environment """
    def __init__(self, state_shape, action_shape, random_seed):
        """Initialise Actors and Critics"""
        self.num_agents = state_shape[0]
        self.obs_size = state_shape[1]                       # size of observation for each agent
        self.state_size = self.num_agents * self.obs_size    # will concatentate observations from each agent to get state (for Critic)
        self.action_size = action_shape[1]
        self.seed = random_seed
        self.maddpg_agent = [None] * self.num_agents
        self.timestep = 0
        for agent_no in range(self.num_agents):
            self.maddpg_agent[agent_no] = Agent(self.num_agents, self.obs_size, self.action_size, self.seed)
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def get_actions(self, state, noise_amplitude):
        actions = np.empty([self.num_agents,self.action_size])
        for agent_no in range(self.num_agents):
            actions[agent_no] = self.maddpg_agent[agent_no].act(state[agent_no], noise_amplitude, add_noise=True) 
        return actions
    
    def reset(self):
        for agent_no in range(self.num_agents):
            self.maddpg_agent[agent_no].reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        #Store state, actions, rewards, next_states, dones in replay buffer
        states = np.reshape(states,(1,48))
        next_states = np.reshape(next_states,(1,48))
        actions = np.reshape(actions,(1,4))
        self.memory.add(states, actions, rewards, next_states, dones)
        self.timestep += 1
        for agent in range(self.num_agents):
            # Sample a random minibatch of S samples from replay buffer, but only every LEARN_EVERY timesteps
            if len(self.memory) > BATCH_SIZE and self.timestep % LEARN_EVERY == 0:
                experiences = self.memory.sample()
                self.learn(experiences, agent, GAMMA)      
        # Soft update of target network parameters
        for agent in range(self.num_agents):
            self.maddpg_agent[agent].soft_update_critic(TAU)
            self.maddpg_agent[agent].soft_update_actor(TAU)
 
    def learn(self,experiences,agent_no, gamma):
        all_states, all_actions, rewards, all_next_states, dones = experiences     #these are already tensors
        obs = torch.narrow(all_states,dim=1,start = 24*agent_no,length = 24)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions from target models
        states0 = torch.narrow(all_states,dim=1, start=0, length=24)
        states1 = torch.narrow(all_states,dim=1, start=24, length=24)
        actions_next0 = self.maddpg_agent[0].actor_target(states0)
        actions_next1 = self.maddpg_agent[1].actor_target(states1)
        all_actions_next = torch.cat((actions_next0, actions_next1), dim=1).to(device)
        # and Q values
        Q_targets_next = self.maddpg_agent[agent_no].critic_target(all_next_states, all_actions_next)
        # Compute Q targets for current states (y_i)
        rewards = torch.narrow(rewards, dim=1, start=agent_no, length=1).to(device)
        dones = torch.narrow(dones, dim=1, start = agent_no, length=1).to(device)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))       
        # Compute critic loss
        Q_expected = self.maddpg_agent[agent_no].critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.maddpg_agent[agent_no].critic_optimizer.zero_grad()
        critic_loss.backward()
        # Try clipping gradient to improve stability
        torch.nn.utils.clip_grad_norm_(self.maddpg_agent[agent_no].critic_local.parameters(), 1)    
        self.maddpg_agent[agent_no].critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred0 = self.maddpg_agent[0].actor_local(states0)
        actions_pred1 = self.maddpg_agent[1].actor_local(states1)
        all_actions_pred = torch.cat((actions_pred0, actions_pred1), dim=1).to(device)      
        actor_loss = -self.maddpg_agent[agent_no].critic_local(all_states, all_actions_pred).mean()
        # Minimize the loss
        self.maddpg_agent[agent_no].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.maddpg_agent[agent_no].actor_optimizer.step()
        
    def save_weights(self):
        for i in range(self.num_agents):
            save_dict = {'actor_params' : self.maddpg_agent[i].actor_local.state_dict(),
                     'critic_params' : self.maddpg_agent[i].critic_local.state_dict()}
            torch.save(save_dict, f"checkpoint_agent_{i}.pth")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, obs_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        critic_input_size = num_agents * (obs_size)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(obs_size, action_size, random_seed).to(device)
        self.actor_target = Actor(obs_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        # Make sure Actor Target Network has same weights as the local Network
        #for target,local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
        #    target.data.copy_(local.data)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(obs_size, action_size, random_seed).to(device)
        self.critic_target = Critic(obs_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # State with Actor Critic Network having same weights as the local Network
        # Not sure this helps...
        #for target,local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
        #    target.data.copy_(local.data)

        # Noise process
        self.noise = OUNoise(action_size, scale=1.0)       #note no random seed. Check scale

        
    def act(self, obs, noise_amplitude, add_noise=True):
        """Returns actions for given state as per current policy."""
        obs = torch.from_numpy(obs).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(obs).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += noise_amplitude * self.noise.noise()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        
    def soft_update_critic(self, tau):
        self.soft_update(self.critic_local, self.critic_target, tau) 
        
    def soft_update_actor(self, tau):
        self.soft_update(self.actor_local, self.actor_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        #self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        np.random.seed(seed)
        #self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)