# 🔥 NeuroFire: AI Autonomous Firefighter Drone

**Two Projects in One:**

1. 🎮 **The Game**: A fully playable Pygame simulation where an AI drone extinguishes fires.
2. 🔬 **The Research Notebook**: A comparative study analyzing **DQN vs PPO vs A2C** algorithms.

An AI-powered firefighting drone simulation trained using **Double Deep Q-Network (Double DQN)** reinforcement learning. The drone learns to navigate a forest environment, detect fires, extinguish them using water, and refill at a lake.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Double DQN Algorithm**: Advanced reinforcement learning with separate policy and target networks
- **Real-time Visualization**: Watch the AI learn with live training metrics and gameplay
- **Smart State Representation**: 17-dimensional state space including danger detection, fire location, lake position, and ammo status
- **Model Persistence**: Save and load trained models for continued training or deployment
- **Graceful Training Control**: Pause training anytime with Ctrl+C and auto-save the model

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
```

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/NeuroFire.git
   cd NeuroFire
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

**Option 1: Using the batch file (Windows)**

```bash
play_neurofire.bat
```

**Option 2: Direct Python execution**

```bash
python main.py
```

## 🔬 Algorithm Comparison

Want to compare different RL algorithms? Check out the comprehensive Jupyter notebook:

**[RL_Algorithm_Comparison_NeuroFire.ipynb](RL_Algorithm_Comparison_NeuroFire.ipynb)** - Compare DQN vs PPO vs A2C

This notebook includes:

- ✅ **Three RL Algorithm Implementations**: DQN, PPO (Proximal Policy Optimization), and A2C (Advantage Actor-Critic)
- ✅ **Side-by-side Training Comparison**: See which algorithm learns fastest
- ✅ **Performance Metrics & Visualizations**: Training curves, loss plots, and reward analysis
- ✅ **Simplified Environment**: Faster experimentation for algorithm testing

> [!NOTE] > **Known Issue**: The PPO agent in the notebook has a minor bug that's documented in the notebook comments. Add `self.actions.append(action.item())` to the `select_action` method before running.

## 🎮 How It Works

### Environment

- **Grid Size**: 640x480 pixels
- **Drone**: Single-point agent that navigates the forest
- **Fires**: Randomly spawn across the forest (avoid lake areas)
- **Lake**: Fixed refill point in the bottom-right corner
- **Water Capacity**: 5 shots maximum

### State Space (17 dimensions)

1. **Danger Detection** (3): Straight, Right, Left
2. **Current Direction** (4): Left, Right, Up, Down
3. **Fire Location** (4): Relative position (Left, Right, Up, Down)
4. **Lake Location** (4): Relative position (Left, Right, Up, Down)
5. **Ammo Status** (1): Out of water (boolean)
6. **Current Position** (1): Head coordinates

### Actions (3)

- **Straight**: Continue in current direction
- **Right Turn**: Turn 90° clockwise
- **Left Turn**: Turn 90° counter-clockwise

### Rewards

| Event                           | Reward          |
| ------------------------------- | --------------- |
| Extinguish fire with water      | +10             |
| Try to fight fire without water | -5              |
| Refill water at lake            | +5              |
| Stay at lake when water is full | -1              |
| Hit boundary wall               | -10 (Game Over) |
| Timeout (100 × score steps)     | -10 (Game Over) |
| Each movement step              | -0.1            |

## 🧠 Architecture

### Double DQN Agent

- **Input Layer**: 17 neurons (state representation)
- **Hidden Layer**: 256 neurons (ReLU activation)
- **Output Layer**: 3 neurons (action Q-values)

### Key Hyperparameters

```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.9              # Discount factor
EPSILON_START = 1.0      # Initial exploration
EPSILON_MIN = 0.01       # Minimum exploration
EPSILON_DECAY = 0.995    # Decay rate per episode
MAX_MEMORY = 100,000     # Replay buffer size
TARGET_UPDATE = 5        # Update target network every 5 episodes
```

## 📊 Training Progress

During training, you'll see:

```
🔥 Starting NeuroFire Training (Double DQN)...
Press Ctrl+C to stop training and save the model.

Game 1 Score 0 Record 0 Epsilon 1.00
Game 2 Score 1 Record 1 Epsilon 0.99
🎉 New Record! Model saved.
Game 3 Score 0 Record 1 Epsilon 0.99
...
```

Real-time plots show:

- **Score per episode** (blue)
- **Average score** (dark blue)
- **Training loss** (red, smoothed)

## 💾 Model Management

### Saving Models

Models are automatically saved to `./model/` when:

- A new record score is achieved → `best_model.pth`
- Training is stopped with Ctrl+C → `final_model.pth`

### Loading a Trained Model

```python
from agent import DoubleDQNAgent

agent = DoubleDQNAgent()
agent.load('best_model.pth')  # Resume training from checkpoint
```

## 📁 Project Structure

```
NeuroFire/
├── main.py              # Training loop and entry point
├── agent.py             # Double DQN agent implementation
├── fire_env.py          # Pygame environment (FireEnv class)
├── model.py             # Neural network architecture
├── helper.py            # Plotting utilities
├── requirements.txt     # Python dependencies
├── play_neurofire.bat   # Windows launcher
├── .gitignore           # Git ignore rules
└── model/               # Saved models (created during training)
    ├── best_model.pth
    └── final_model.pth
```

## 🛠️ Development

### Code Quality

This project follows best practices:

- ✅ Type hints and docstrings
- ✅ Modular architecture (separate concerns)
- ✅ No dead code
- ✅ Proper error handling
- ✅ Configurable hyperparameters

### Testing

Run a quick test:

```python
python -c "import torch; import pygame; import matplotlib; print('All dependencies OK!')"
```

## 📈 Performance Tips

1. **Speed up training**: Increase `SPEED` in `fire_env.py` (line 30)
2. **Improve exploration**: Adjust `epsilon_decay` in `agent.py`
3. **Larger network**: Increase `hidden_size` in `DoubleDQNAgent.__init__()`
4. **More memory**: Increase `MAX_MEMORY` in `agent.py`

## 🐛 Troubleshooting

**Pygame window doesn't appear:**

- Ensure your display is properly configured
- Try running with `python -u main.py` for unbuffered output

**Training is too slow:**

- Increase `SPEED` in `fire_env.py`
- Reduce visualization frequency in `helper.py`

**Memory issues:**

- Reduce `MAX_MEMORY` in `agent.py`
- Reduce `BATCH_SIZE` in `agent.py`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classic Snake game AI projects
- Uses PyTorch for deep learning
- Pygame for environment visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ and 🔥 by AI Enthusiasts**
