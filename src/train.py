import os
import numpy as np

from dataset import load_sequence, load_frame, load_groundtruth
from ppo_agent import PPO
from feature_extractor import ResNetExtractor
from tracker_utils import apply_action, clamp_box, iou

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TrackingEnv:
    """Environment for RL-based visual tracking."""
    
    def __init__(self, dataset_path, sequences):
        self.dataset_path = dataset_path
        self.sequences = sequences
        self.action_space = 6
        self.state_dim = 516  # 512 features + 4 box coordinates
        
        self.feature_extractor = ResNetExtractor().eval()
    
    def reset(self):
        """Start new episode with random sequence."""
        self.seq = np.random.choice(self.sequences)
        self.frames, self.gt = load_sequence(self.dataset_path, self.seq)
        self.idx = 0
        self.box = self.gt[0].copy()
        return self._get_state()
    
    def _get_state(self):
        """Extract state vector: features + normalized box."""
        img = load_frame(self.frames[self.idx])
        self.box = clamp_box(self.box, img.shape)
        
        x, y, w, h = self.box.astype(int)
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            crop = np.zeros((128, 128, 3), dtype=np.uint8)
        
        with torch.no_grad():
            feat = self.feature_extractor(crop).detach().cpu().numpy()
        
        return np.concatenate([feat, self.box / 500.0]).astype(np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done."""
        img = load_frame(self.frames[self.idx])
        self.box = clamp_box(self.box, img.shape)
        
        # Apply action
        self.box = apply_action(self.box, action)
        self.box = clamp_box(self.box, img.shape)
        
        self.idx += 1
        
        if self.idx >= len(self.frames) - 1:
            self.idx = len(self.frames) - 1
            return self._get_state(), 0.0, True
        
        # Compute reward using IoU
        gt_idx = min(self.idx, len(self.gt) - 1)
        reward = iou(self.box, self.gt[gt_idx])
        
        return self._get_state(), reward, False

def train():
    """Main training loop."""
    DATASET_PATH = "data/OTB100"
    
    # Get all sequences
    sequences = [
        s for s in sorted(os.listdir(DATASET_PATH))
        if os.path.isdir(os.path.join(DATASET_PATH, s))
    ]
    print(f"Found {len(sequences)} sequences")
    
    env = TrackingEnv(DATASET_PATH, sequences)
    agent = PPO(state_dim=env.state_dim, action_dim=env.action_space)
    
    max_episodes = 1000
    max_steps = 500
    update_interval = 1000
    
    # Experience buffer
    states, actions, log_probs, rewards = [], [], [], []
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action, log_prob, _ = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            
            # Update policy periodically
            if len(states) >= update_interval:
                loss = agent.update(states, actions, log_probs, rewards)
                states, actions, log_probs, rewards = [], [], [], []
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode:3d} | Reward: {episode_reward:6.2f} | Avg Reward: {avg_reward:6.2f}")
        
        # Save checkpoint
        if episode % 100 == 0 and episode > 0:
            agent.save(f"checkpoint_ep{episode}.pth")
            print(f"  -> Saved checkpoint_ep{episode}.pth")
    
    # Save final model
    agent.save("ppo_tracker_best.pth")
    print("\nTraining complete! Model saved to ppo_tracker_best.pth")

if __name__ == "__main__":
    train()