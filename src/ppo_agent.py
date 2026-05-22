import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    """PPO agent with policy network for discrete actions."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        super().__init__()
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
    def select_action(self, state):
        """Sample action from policy distribution."""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), probs.detach()
    
    def update(self, states, actions, log_probs_old, rewards):
        """Update policy using PPO clipped objective."""
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        log_probs_old = torch.FloatTensor(np.array(log_probs_old)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        
        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards.cpu().numpy()):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(np.array(returns)).to(device)
        advantages = (returns - r-eturns.mean()) / (returns.std() + 1e-8)
        
        # Compute new log probabilities
        logits = self.policy_net(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        
        # PPO clip loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        return policy_loss.item()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.eval()