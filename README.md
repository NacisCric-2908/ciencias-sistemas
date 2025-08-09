# Predator-Prey Adaptive Agent Simulation — Workshop 3

This repository contains the theoretical and architectural design for an adaptive simulation system involving a **predator** and a **prey**, each modeled as autonomous agents capable of learning and adaptation through **reinforcement learning** and **cybernetic control**.

> ⚠️ This project focuses on **system design** and not implementation. The learning pipelines and simulation framework are to be developed in subsequent stages.

---

## 🧠 Project Objective

Design a two-agent competitive simulation where:

- The **Predator** learns to **capture** the prey efficiently.
- The **Prey** learns to **evade** the predator as long as possible.

Both agents operate under **partially observable environments**, use **feedback mechanisms** to regulate movement, and are trained using **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.

---

## 🧩 System Overview

| Component          | Description |
|-------------------|-------------|
| `Agent`           | Base class for shared properties (position, movement, sensors). |
| `Predator`        | Adaptive agent trained to intercept prey using gradient-following and feedback-based speed. |
| `Prey`            | Evasive agent trained to maximize survival time using probabilistic escape learning. |
| `Maze`            | 2D grid environment with walls, traps, and aroma diffusion. |
| `Aroma Matrix`    | Dynamic scent grid that decays over time, used by the predator for spatial tracking. |

---

## 🔁 Feedback and Dynamics

- Agents follow discrete-time motion equations:
  - Predator: `πₚ(t+1) = πₚ(t) + Δt ⋅ rₚ(t) ⋅ vₚ(t)`
  - Prey: `πₕ(t+1) = πₕ(t) + Δt ⋅ rₕ(t) ⋅ uₕ(t)`
- Speed and direction are modulated by:
  - Prey proximity (for predator).
  - Predator threat and trap detection (for prey).
- Feedback loops ensure real-time regulation and decision adaptation.

---

## 📐 Learning Architecture

- **Algorithm Used:** `MADDPG`
- **Training Style:** Centralized training with decentralized execution.
- **Observations:**
  - Local grid (5×5) around agent.
  - Aroma matrix intensity.
  - Trap proximity.
- **Rewards:**
  - Predator: `+10` for capture, `-0.1` per step.
  - Prey: `+0.1` per step alive, `-5` if caught.

---

## 📊 Evaluation Strategy

- Episode reward tracking
- Capture/survival rate
- Phase diagram visualization
- Emergent behavior analysis
- Generalization to new map layouts and conditions

---

