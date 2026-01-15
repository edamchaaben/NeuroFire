import torch
import random
import numpy as np
from collections import deque
from fire_env import FireEnv, Point, Direction
from model import Linear_QNet, QTrainer
import torch.optim as optim
import torch.nn.functional as F

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

class ReplayBuffer:
    def __init__(self, capacity=MAX_MEMORY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    def __init__(self, input_size=16, hidden_size=256, output_size=3):
        self.n_games = 0
        self.epsilon = 1.0 # Decay this over time
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.9 # Discount rate
        
        self.memory = ReplayBuffer()
        
        # Double DQN: Policy Net (Action Selection) & Target Net (Evaluation)
        self.policy_net = Linear_QNet(input_size, hidden_size, output_size)
        self.target_net = Linear_QNet(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync initially
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()

    def get_state(self, game):
        # Same state logic as before
        head = game.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Fire Location 
            game.food.x < game.head.x,  # Fire is Left
            game.food.x > game.head.x,  # Fire is Right
            game.food.y < game.head.y,  # Fire is Up
            game.food.y > game.head.y, # Fire is Down

            # Lake Location
            game.lake.x < game.head.x, 
            game.lake.x > game.head.x, 
            game.lake.y < game.head.y, 
            game.lake.y > game.head.y,
            
            # Ammo Status
            game.ammo == 0
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.policy_net(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # 1. Action Selection (Policy Net)
        # actions is one-hot [0, 1, 0], we need index [1]
        action_indices = torch.argmax(actions, dim=1).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, action_indices).squeeze()

        # 2. Target Evaluation (Target Net) - Double DQN
        with torch.no_grad():
            # Select best action from Policy Net
            next_actions = torch.argmax(self.policy_net(next_states), dim=1).unsqueeze(1)
            # Evaluate that action using Target Net
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 3. Loss & Backprop
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename='best_model.pth'):
        """
        Save the agent's policy network to a file.
        """
        import os
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_path = os.path.join(model_folder_path, filename)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon
        }, file_path)

    def load(self, filename='best_model.pth'):
        """
        Load a previously saved agent.
        """
        import os
        file_path = os.path.join('./model', filename)
        
        if not os.path.exists(file_path):
            print(f"Model file {file_path} not found.")
            return False
        
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint['n_games']
        self.epsilon = checkpoint['epsilon']
        
        print(f"Model loaded from {file_path}")
        print(f"Resuming from game {self.n_games}, epsilon {self.epsilon:.2f}")
        return True
