# 🧠 Predator–Prey Autonomous Adaptive Agent System

This repository contains the design and implementation of an adaptive autonomous multi-agent system that simulates the interaction between a predator and a prey in a discrete environment. The project was developed as part of two academic workshops for the Systems Sciences course (2025-I).

## 📁 Contents

- `Workshop_1.pdf`: System requirements and functional design document.
- `Workshop_2.pdf`: Dynamic system analysis, mathematical modeling, simulation, and stability validation.
- Source code (coming soon).
- Simulation data and visualizations (optional).

---

## 🎯 Objective

To build a simulated cyber-physical system where:
- A **predator** learns to hunt.
- A **prey** learns to evade using the environment.
- Both agents adapt through *reinforcement learning*.

---

## 🧪 Key Features

### 🧬 Workshop 1 – System Design

- **Sensors**:
  - Prey: proximity sensor (5x5) and environmental sensor (7x7).
  - Predator: prey detection sensor (10x10) and environmental sensor (6x6).
- **Actuators**: movement control, environmental learning, and trap detection.
- **Reward functions**:
  - Predator: +10 for capture, +0.05 when close, penalties for delays or failed hunts.
  - Prey: +0.1 for surviving, +0.5 for escaping, penalties for being caught or trapped.
- **Learning**:
  - Q-learning and Deep Q-Networks (DQN) via Stable-Baselines3.
  - Cybernetic feedback loops for strategy adaptation.

### 🧮 Workshop 2 – Dynamic System Analysis & Simulation

- **Mathematical Model**:
  - Adaptive predator speed based on distance.
  - Increasing evasion probability for the prey over time.
- **Stability**:
  - Equilibrium points: capture (distance = 0) or infinite evasion.
  - Local analysis using the Jacobian matrix.
  - Lyapunov function based on total kinetic energy.
- **Simulation**:
  - 10×10 grid, up to 1000 steps.
  - Visualization of trajectories and performance metrics.
- **Results**:
  - 70% of runs result in capture before step 200.
  - Energy function decreases consistently, supporting system stability.

---

## 🛠️ Requirements

- Python 3.10+
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyTorch](https://pytorch.org/)
- Matplotlib (for visualization)

Quick install with pip:

```bash
pip install gymnasium[all] stable-baselines3 torch matplotlib
