# Predator-Prey Simulation with Adaptive AI Agents

## 📖 Overview

This project implements an adaptive predator-prey simulation in a 2D grid environment, featuring:
- Autonomous agents with reactive/proactive behaviors
- Dynamic environmental interactions (traps, smell trails)
- Machine learning-based decision making (XGBoost classifier)
- Reinforcement learning compatibility via PettingZoo API
- Probabilistic trap evasion mechanics
- Multi-modal predator behavior (patrol/hunting)

Based on research in adaptive systems, reinforcement learning, and biologically-inspired AI.

## 🚀 Key Features

- **Hybrid AI Architecture**:
  - Rule-based prey with learning evasion
  - ML-driven predator with behavior switching
- **Environmental Dynamics**:
  - Randomized trap placement
  - Decaying smell trails
  - Spatial constraints
- **Reinforcement Learning Ready**:
  - PettingZoo-compatible interface
  - Modular observation/action spaces
  - Reward logging and visualization

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/predator-prey-simulation.git
   cd predator-prey-simulation
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Download assets (sprites, tiles) and place in src/Visualization/assets

## 🎮 Usage

1. Training Mode:
   ```bash
   python prey_predator_Env_train.py
2. Evaluation Mode (with visualization):
   ```bash
   python prey_predator_Env_eval.py

Key Parameters (in constant_variables.py)
- Agent speeds and sizes
- Smell evaporation rates
- Trap evasion probabilities
- Movement constraints
- Rendering options

## 📂 Project Structure
predator-prey-simulation/
├── src/
│   ├── Visualization/          # Rendering assets
│   ├── Agent.py               # Base agent class
│   ├── Prey.py                # Prey behaviors
│   ├── Predator.py            # Predator AI
│   ├── Maze.py                # Environment generation
│   ├── utils.py               # Helper functions
│   ├── constant_variables.py  # Simulation parameters
│   ├── prey_predator_Env.py   # Core environment
│   ├── prey_predator_Env_rllib.py  # RLlib integration
│   ├── prey_predator_Env_train.py  # Training script
│   └── prey_predator_Env_eval.py   # Evaluation script

## 📊 Results
- Performance metrics from 1000-episode evaluation:
- Prey Survival: 78.6±18.4 steps (late episodes)
- Predator Success: 63.2% capture rate
- Trap Evasion: 87.2% success (late episodes)
- Hunting Accuracy: 82.4% optimal activation

## 📚 Theoretical Foundation
The system combines:
  - Control Theory: Stability analysis of motion dynamics
  - Machine Learning: XGBoost for behavior switching
  - Reinforcement Learning: Compatible with PPO/DQN
  - Biological Inspiration: Smell trails, adaptive evasion

## 💡 Future Work
- Deep RL integration (PPO/DQN)
- Multi-agent coordination
- Continuous action spaces
- Enhanced visualization tools
- Curriculum learning setups
