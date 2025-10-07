# Reinforcement Learning for Autonomous Driving in AirSim (PPO)

End-to-end reinforcement learning pipeline to train an autonomous car in Microsoft AirSim, using a custom Gym environment with depth image + vehicle velocity, reward shaping, and PPO training/evaluation.

## Introduction

This case study applies RL and reproducible experimentation to autonomous navigation in simulation.  
We train and evaluate a driving policy that learns from depth perception and speed feedback to avoid collisions and make forward progress, with a codebase designed for quick iteration (sensors, rewards, maps, and hyperparameters).

When deploying RL policies, two risks exist:
1. Unsafe behavior → collisions / off-track episodes  
2. Overly conservative behavior → low progress / timeouts

Careful reward design and evaluation help balance these trade-offs.

## Business Understanding

For AV research, simulation enables fast, safe iteration on sensing and control.  
We use AirSim’s Car mode to test how sensor choices (depth + velocity) and reward shaping influence driving behavior, aiming to reduce collisions and improve stable progress, insights that transfer to future sim-to-real studies.

## Business Objectives

- Learn a collision-avoiding policy that maintains forward motion  
- Provide a modular, reproducible training stack (env, config, scripts)  
- Identify design levers (reward weights, observation set, action ranges) that most improve stability

## Analytical Approach

1. **Environment & Setup**  
   - AirSim Car SimMode; front DepthPerspective camera via `settings.json`  
   - Custom Gym env (`airsim_rl_env.py`) exposing a dict observation: depth frame + velocity  
   - Continuous actions: steering, throttle, brake

2. **Reward Design**  
   - Positive: forward progress, staying near lane center  
   - Negative: lateral drift/oscillation, unsafe speed; hard penalty on collision (terminate)  
   - Optional anti-loop tweaks after respawn

3. **Modeling (PPO)**  
   - Stable-Baselines3 PPO (`MultiInputPolicy`)  
   - Sensible defaults for learning rate, n_steps, batch_size, entropy/clip ranges  
   - Vectorized environment compatible; TensorBoard logging

4. **Training & Evaluation**  
   - Train for fixed timesteps; save checkpoints and final model  
   - Track reward curves, episode length, collision counts  
   - Optional eval script to run deterministic rollouts

5. **Ablations (optional)**  
   - Depth-only vs depth+velocity  
   - Reward coefficients and action-range sensitivity  
   - Image resolution trade-offs

## Tools & Libraries

- Python (Jupyter/CLI)  
- Microsoft AirSim, gym (0.26.x)  
- stable-baselines3 (PPO), PyTorch, TensorBoard  
- NumPy, OpenCV (optional for frame ops)

## Environment & Observations

- **Observations:** single-channel depth image (e.g., 84×84×1) + scalar vehicle speed  
- **Actions:** continuous steering (−1…1), throttle (0…1), brake (0…1)  
- **Episodes:** terminate on collision or time limit; env handles respawn/reset  
- **Config:** `settings.json` enables Car mode and the required camera

## Key Insights

This workflow typically surfaces:
- Reward shaping that rewards progress and penalizes collisions stabilizes training  
- Including velocity often reduces throttle oscillations vs image-only policies  
- Proper action bounds and image preprocessing (resize/crop/normalize) improve convergence  
- PPO provides smoother learning for continuous control than value-only baselines

(Exact metrics/curves appear in the write-up/slides or TensorBoard logs.)

## How to Run

1) Configure AirSim:
- Windows: `%USERPROFILE%\Documents\AirSim\settings.json`  
- Linux/macOS: `~/Documents/AirSim/settings.json`

2) Install dependencies (example):
- pip install airsim gym==0.26.2 numpy stable-baselines3==2.3.0 torch tensorboard opencv-python

3) Start training:
- python train_rl.py

4) Monitor training:
- tensorboard --logdir runs

## Files in this Repository

| File Name | Description |
| --- | --- |
| `train_rl.py` | PPO training script (Stable-Baselines3), logging and checkpoints |
| `airsim_rl_env.py` | Custom Gym environment (depth + velocity), controls and resets |
| `settings.json` | AirSim config (SimMode="Car", DepthPerspective camera) |
| `Autonomous Vehicle Navigation Using Reinforcement Learning in AirSim.pdf` | Project write-up: approach, experiments, results |
| `Reinforcement Learning Based Autonomous Driving in AirSim.pptx` | Slide deck for stakeholders |

## Author

**Aakash Maskara**  
*M.S. Robotics & Autonomy, Drexel University*  
Robotics | Reinforcement Learning | Autonomous Systems

[LinkedIn](https://linkedin.com/in/aakashmaskara) • [GitHub](https://github.com/aakashmaskara)
