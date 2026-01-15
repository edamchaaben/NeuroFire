"""
RL Algorithms Comparison: DQN vs PPO vs A2C
A comprehensive comparison framework for NeuroFire project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Tuple, List, Dict
import time
from dataclasses import dataclass
import pandas as pd

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AlgorithmMetrics:
    """Store metrics for each algorithm"""
    name: str
    episodes: List[int]
    rewards: List[float]
    losses: List[float]
    mean_rewards: List[float]
    training_times: List[float]
    success_rate: float
    total_time: float
    convergence_episode: int = -1

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based learning"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Policy network for actor-critic methods (PPO/A2C)"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        policy = self.policy(shared_features)
        value = self.value(shared_features)
        return policy, value


# ============================================================================
# REPLAY BUFFER
# ============================================================================

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity: int = 100_000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.memory)


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, 
                 input_size: int = 11,
                 output_size: int = 3,
                 hidden_size: int = 256,
                 lr: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        
        # Network and training parameters
        self.policy_net = DQNNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net = DQNNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.memory = ReplayBuffer()
        self.batch_size = 64
        
        # Metrics
        self.losses = []
        self.rewards = []
        self.mean_rewards = []
        self.q_values = []
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.output_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self) -> float:
        """Training step using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Compute Q(s,a) - the current estimate
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s') - the value of next state using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(q_values.squeeze(), target_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        if len(self.memory) % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.losses.append(loss.item())
        return loss.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        self.memory.push(state, action, reward, next_state, done)


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self,
                 input_size: int = 11,
                 output_size: int = 3,
                 hidden_size: int = 256,
                 lr: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 epochs: int = 3):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyNetwork(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        # Storage for trajectories
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
        
        self.losses = []
        self.rewards = []
        self.mean_rewards = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action and return log probability"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.network(state_tensor)
        
        action = np.random.choice(policy.shape[1], p=policy.detach().cpu().numpy()[0])
        log_prob = torch.log(policy[0, action])
        
        return action, log_prob
    
    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        rewards = self.trajectory['rewards']
        values = self.trajectory['values']
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def train_step(self):
        """PPO training step"""
        if not self.trajectory['states']:
            return 0.0
        
        # Prepare data
        states = torch.FloatTensor(np.array(self.trajectory['states'])).to(self.device)
        actions = torch.LongTensor(np.array(self.trajectory['actions'])).to(self.device)
        old_log_probs = torch.stack(self.trajectory['log_probs']).to(self.device)
        
        # Compute advantages
        advantages, returns = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        
        # PPO epochs
        for _ in range(self.epochs):
            policies, values = self.network(states)
            
            # Get new log probabilities
            log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)).squeeze())
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.losses.append(total_loss / self.epochs)
        
        # Clear trajectory
        self.trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
        
        return total_loss / self.epochs
    
    def store_transition(self, state, action, reward, value, log_prob):
        """Store transition"""
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)
        self.trajectory['values'].append(value.item())
        self.trajectory['log_probs'].append(log_prob)


# ============================================================================
# A2C AGENT
# ============================================================================

class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(self,
                 input_size: int = 11,
                 output_size: int = 3,
                 hidden_size: int = 256,
                 lr: float = 0.0003,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyNetwork(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.losses = []
        self.rewards = []
        self.mean_rewards = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return policy, value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        policy, value = self.network(state_tensor)
        action = np.random.choice(policy.shape[1], p=policy.detach().cpu().numpy()[0])
        
        return action, policy[0, action], value
    
    def train_step(self, state, action, reward, next_state, done):
        """A2C training step (on-policy, single step)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get current and next values
        _, current_value = self.network(state_tensor)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
        
        # Compute TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value.item()
        
        # Recompute for gradient
        policy, value = self.network(state_tensor)
        
        # Advantage
        advantage = target - value.item()
        
        # Policy loss
        log_prob = torch.log(policy[0, action])
        policy_loss = -log_prob * advantage
        
        # Entropy bonus
        entropy = -(policy * torch.log(policy + 1e-8)).sum()
        
        # Value loss
        value_loss = (value.squeeze() - target) ** 2
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        self.losses.append(total_loss.item())
        return total_loss.item()


# ============================================================================
# SIMPLE TEST ENVIRONMENT
# ============================================================================

class SimpleTestEnv:
    """Simplified environment for algorithm comparison"""
    
    def __init__(self):
        self.state = np.random.randn(11)
        self.steps = 0
        self.max_steps = 100
    
    def reset(self):
        self.state = np.random.randn(11)
        self.steps = 0
        return self.state
    
    def step(self, action):
        self.steps += 1
        
        # Simple reward based on action
        reward = np.random.randn()
        if action == 0:
            reward += 0.5
        
        # Random next state
        self.state = np.random.randn(11)
        
        done = self.steps >= self.max_steps
        
        return self.state, reward, done
    
    def render(self):
        pass


# ============================================================================
# COMPARISON FRAMEWORK
# ============================================================================

class AlgorithmComparison:
    """Framework for comparing RL algorithms"""
    
    def __init__(self, episodes: int = 100, runs: int = 3):
        self.episodes = episodes
        self.runs = runs
        self.metrics: Dict[str, AlgorithmMetrics] = {}
    
    def run_dqn(self, env) -> AlgorithmMetrics:
        """Run DQN agent"""
        print("\n🎯 Training DQN Agent...")
        agent = DQNAgent()
        
        episodes_list = []
        rewards_list = []
        mean_rewards = []
        losses_list = []
        start_time = time.time()
        
        episode_rewards = deque(maxlen=100)
        
        for episode in range(self.episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            rewards_list.append(total_reward)
            mean_rewards.append(np.mean(episode_rewards))
            episodes_list.append(episode)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Mean Reward: {mean_rewards[-1]:.2f}")
        
        total_time = time.time() - start_time
        
        # Find convergence episode
        convergence_episode = self._find_convergence_episode(mean_rewards)
        
        metrics = AlgorithmMetrics(
            name="DQN",
            episodes=episodes_list,
            rewards=rewards_list,
            losses=agent.losses,
            mean_rewards=mean_rewards,
            training_times=[],
            success_rate=self._calculate_success_rate(rewards_list),
            total_time=total_time,
            convergence_episode=convergence_episode
        )
        
        return metrics
    
    def run_ppo(self, env) -> AlgorithmMetrics:
        """Run PPO agent"""
        print("\n🎯 Training PPO Agent...")
        agent = PPOAgent()
        
        episodes_list = []
        rewards_list = []
        mean_rewards = []
        losses_list = []
        start_time = time.time()
        
        episode_rewards = deque(maxlen=100)
        
        for episode in range(self.episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                # Store for PPO
                with torch.no_grad():
                    _, value = agent.network(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
                
                agent.store_transition(state, action, reward, value.squeeze(), log_prob)
                
                total_reward += reward
                state = next_state
            
            # Train on trajectory
            loss = agent.train_step()
            
            episode_rewards.append(total_reward)
            rewards_list.append(total_reward)
            mean_rewards.append(np.mean(episode_rewards))
            episodes_list.append(episode)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Mean Reward: {mean_rewards[-1]:.2f}")
        
        total_time = time.time() - start_time
        convergence_episode = self._find_convergence_episode(mean_rewards)
        
        metrics = AlgorithmMetrics(
            name="PPO",
            episodes=episodes_list,
            rewards=rewards_list,
            losses=agent.losses,
            mean_rewards=mean_rewards,
            training_times=[],
            success_rate=self._calculate_success_rate(rewards_list),
            total_time=total_time,
            convergence_episode=convergence_episode
        )
        
        return metrics
    
    def run_a2c(self, env) -> AlgorithmMetrics:
        """Run A2C agent"""
        print("\n🎯 Training A2C Agent...")
        agent = A2CAgent()
        
        episodes_list = []
        rewards_list = []
        mean_rewards = []
        losses_list = []
        start_time = time.time()
        
        episode_rewards = deque(maxlen=100)
        
        for episode in range(self.episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _, _ = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                loss = agent.train_step(state, action, reward, next_state, done)
                losses_list.append(loss)
                
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            rewards_list.append(total_reward)
            mean_rewards.append(np.mean(episode_rewards))
            episodes_list.append(episode)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Mean Reward: {mean_rewards[-1]:.2f}")
        
        total_time = time.time() - start_time
        convergence_episode = self._find_convergence_episode(mean_rewards)
        
        metrics = AlgorithmMetrics(
            name="A2C",
            episodes=episodes_list,
            rewards=rewards_list,
            losses=losses_list,
            mean_rewards=mean_rewards,
            training_times=[],
            success_rate=self._calculate_success_rate(rewards_list),
            total_time=total_time,
            convergence_episode=convergence_episode
        )
        
        return metrics
    
    def run_comparison(self) -> Dict[str, AlgorithmMetrics]:
        """Run all algorithms"""
        env = SimpleTestEnv()
        
        self.metrics['DQN'] = self.run_dqn(env)
        self.metrics['PPO'] = self.run_ppo(env)
        self.metrics['A2C'] = self.run_a2c(env)
        
        return self.metrics
    
    @staticmethod
    def _find_convergence_episode(mean_rewards: List[float], threshold: float = 0.95) -> int:
        """Find episode where algorithm converged"""
        if not mean_rewards:
            return -1
        
        target = threshold * np.max(mean_rewards[-20:])
        for i, reward in enumerate(mean_rewards):
            if reward >= target:
                return i
        
        return -1
    
    @staticmethod
    def _calculate_success_rate(rewards: List[float]) -> float:
        """Calculate success rate (% of positive rewards)"""
        if not rewards:
            return 0.0
        return sum(1 for r in rewards if r > 0) / len(rewards)


# ============================================================================
# VISUALIZATION
# ============================================================================

class ComparisonVisualizer:
    """Visualize algorithm comparison results"""
    
    @staticmethod
    def plot_comparison(metrics: Dict[str, AlgorithmMetrics], save_path: str = None):
        """Create comprehensive comparison plots"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        colors = {'DQN': '#2E86AB', 'PPO': '#A23B72', 'A2C': '#F18F01'}
        
        # 1. Learning Curves (Rewards)
        ax1 = fig.add_subplot(gs[0, :2])
        for name, metric in metrics.items():
            ax1.plot(metric.episodes, metric.mean_rewards, 
                    label=name, linewidth=2, color=colors[name])
        ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Mean Reward (100 episodes)', fontsize=11, fontweight='bold')
        ax1.set_title('Learning Curves: Algorithm Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Success Rate Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        names = list(metrics.keys())
        success_rates = [metrics[name].success_rate for name in names]
        bars = ax2.bar(names, success_rates, color=[colors[n] for n in names], alpha=0.7)
        ax2.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
        ax2.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1])
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Training Loss
        ax3 = fig.add_subplot(gs[1, 0])
        for name, metric in metrics.items():
            if metric.losses:
                ax3.plot(metric.losses[:200], label=name, linewidth=1.5, color=colors[name], alpha=0.8)
        ax3.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax3.set_title('Training Loss (First 200 steps)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence Speed
        ax4 = fig.add_subplot(gs[1, 1])
        names = list(metrics.keys())
        convergence_episodes = [metrics[name].convergence_episode if metrics[name].convergence_episode != -1 
                               else metrics[name].episodes[-1] for name in names]
        bars = ax4.bar(names, convergence_episodes, color=[colors[n] for n in names], alpha=0.7)
        ax4.set_ylabel('Episodes to Convergence', fontsize=11, fontweight='bold')
        ax4.set_title('Convergence Speed', fontsize=13, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Training Time
        ax5 = fig.add_subplot(gs[1, 2])
        names = list(metrics.keys())
        times = [metrics[name].total_time for name in names]
        bars = ax5.bar(names, times, color=[colors[n] for n in names], alpha=0.7)
        ax5.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax5.set_title('Total Training Time', fontsize=13, fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 6. Reward Distribution
        ax6 = fig.add_subplot(gs[2, :])
        positions = []
        data = []
        labels = []
        for i, (name, metric) in enumerate(metrics.items()):
            positions.append(i)
            data.append(metric.rewards)
            labels.append(name)
        
        bp = ax6.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2))
        
        for patch, name in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.7)
        
        ax6.set_xticklabels(labels)
        ax6.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
        ax6.set_title('Reward Distribution', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('RL Algorithms Comparison: DQN vs PPO vs A2C', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Comparison plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def print_summary(metrics: Dict[str, AlgorithmMetrics]):
        """Print detailed comparison summary"""
        
        print("\n" + "="*80)
        print("🎯 RL ALGORITHMS COMPARISON SUMMARY".center(80))
        print("="*80)
        
        # Create DataFrame for easy comparison
        data = []
        for name, metric in metrics.items():
            data.append({
                'Algorithm': name,
                'Final Mean Reward': f"{metric.mean_rewards[-1]:.4f}",
                'Max Mean Reward': f"{max(metric.mean_rewards):.4f}",
                'Success Rate': f"{metric.success_rate:.2%}",
                'Convergence Episode': metric.convergence_episode if metric.convergence_episode != -1 else 'N/A',
                'Total Time (s)': f"{metric.total_time:.2f}",
                'Total Steps': len(metric.rewards)
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        
        # Detailed analysis
        print("\n" + "-"*80)
        print("📊 DETAILED METRICS")
        print("-"*80)
        
        for name, metric in metrics.items():
            print(f"\n{name} Agent:")
            print(f"  • Episodes trained: {len(metric.episodes)}")
            print(f"  • Final reward: {metric.mean_rewards[-1]:.4f}")
            print(f"  • Best reward: {max(metric.mean_rewards):.4f}")
            print(f"  • Success rate: {metric.success_rate:.2%}")
            print(f"  • Convergence: Episode {metric.convergence_episode}" if metric.convergence_episode != -1 else f"  • Convergence: Did not converge")
            print(f"  • Training time: {metric.total_time:.2f}s")
            print(f"  • Avg time per episode: {metric.total_time/len(metric.episodes):.4f}s")
        
        # Comparison insights
        print("\n" + "-"*80)
        print("💡 KEY INSIGHTS")
        print("-"*80)
        
        # Best in each category
        best_reward = max(metrics.items(), key=lambda x: max(x[1].mean_rewards))
        best_speed = min(metrics.items(), key=lambda x: x[1].total_time)
        best_convergence = min([(name, m.convergence_episode) for name, m in metrics.items() 
                               if m.convergence_episode != -1], key=lambda x: x[1], default=('N/A', -1))
        
        print(f"\n✨ Best Final Performance: {best_reward[0]} ({best_reward[1].mean_rewards[-1]:.4f})")
        print(f"⚡ Fastest Training: {best_speed[0]} ({best_speed[1].total_time:.2f}s)")
        if best_convergence[0] != 'N/A':
            print(f"🎯 Fastest Convergence: {best_convergence[0]} (Episode {best_convergence[1]})")
        
        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting RL Algorithm Comparison Framework...")
    print("Comparing DQN, PPO, and A2C algorithms\n")
    
    # Run comparison
    comparison = AlgorithmComparison(episodes=100, runs=1)
    metrics = comparison.run_comparison()
    
    # Visualize results
    visualizer = ComparisonVisualizer()
    visualizer.plot_comparison(metrics, save_path='comparison_results.png')
    visualizer.print_summary(metrics)
    
    print("\n✅ Comparison complete!")
    print(f"📊 Results saved to comparison_results.png")
