# TD3-AHLR-LC
This repository contains the code implementation of the TD3-based lane change control model proposed in the paper *Multi-Objective Deep Reinforcement Learning-Driven Human-Like Lane-Changing Model in a Dynamic Interaction-Aware Connected Environment* (Hongru Yu et al.).
Note: This code is a demo version, and its content may be updated in the future.


## Project Overview
This code implements an adaptive lane change control model using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, integrated with:
- A multi-dimensional reward mechanism (comfort + safety + efficiency)
- Human-like lane change completion time (LCT) guiding reward (based on LightGBM-estimated Gaussian kernel density)
- A hybrid traffic flow environment (natural driving trajectories + Intelligent Driver Model)


## File Structure 
| File               | Component                                                                 | 
| `config.py`         | Model/environment hyperparameters                                                             |
| `lgb.py`            | Extended LightGBM for LCT Gaussian kernel density estimation                                  | 
| `main.py`           | End-to-end training/test pipeline                                                             | 
| `simulation_env.py` | Hybrid traffic flow environment (natural trajectories + IDM)                                  | 
| `tf2rl.py`          | TD3 algorithm for lane change control (extended for longitudinal/lateral acceleration control)| 
| `traj_processing.py`| Natural driving trajectory preprocessing                                                     | 
| `utils.py`          | Auxiliary tools                                                                               | 


## Environment Setup
Install required dependencies (consistent with the paper's experimental environment):
pip install tensorflow==2.15.0 lightgbm==4.6.0 numpy pandas tqdm joblib

## Usage
-Configure Parameters
Modify hyperparameters (TD3 settings, environment thresholds, LightGBM hyperparameters, etc.) in config.py (default values align with the paper's experimental setup).
-Prepare Trajectories
Place raw natural driving trajectory data in the directory specified in config.py (the code will automatically preprocess trajectories via traj_processing.py).
-Run the End-to-End Pipeline
Execute the full training and evaluation workflow: python main.py

The script executes the following steps sequentially:
-Trajectory Preprocessing: traj_processing.py screens valid lane change trajectories and constructs feature samples.
-Environment Initialization: simulation_env.py builds the hybrid traffic environment (natural trajectories + IDM).
-TD3 Agent Training: tf2rl.py trains the TD3 agent with the adaptive human-like reward mechanism (guided by lgb.py's LCT estimation).
-Result Evaluation: Calculates key metrics (TTC < 3s ratio, lane change efficiency, collision rate) as reported in the paper.

## Details
-TD3 Agent (tf2rl.py)
Implements the TD3 algorithm extended for lane change control:
Dual Critic networks to mitigate overestimation bias in value estimation.
Asynchronous policy updates (2-step delay) to stabilize training.
Joint control of longitudinal/lateral accelerations to match lane change dynamics.
Adaptive action noise exploration (noise decay during training to balance exploration/exploitation).
-Hybrid Traffic Environment (simulation_env.py)
Implements the high-fidelity traffic environment:
Fuses natural driving trajectory data to simulate realistic lane change scenarios.
Integrates the Intelligent Driver Model (IDM) to simulate rear vehicle following behavior.
Real-time collision detection (Time-to-Collision (TTC) threshold = 3.001s).
Multi-dimensional reward calculation:
Comfort: Jerk/acceleration smoothness penalties.
Safety: TTC-based reward to avoid dangerous behaviors.
Efficiency: Lane change completion reward with human-like LCT guidance.
-LCT Gaussian Kernel Density Estimation (lgb.py)
Implements the human-like LCT adaptive reward:
Trains LightGBM models to predict LCT's mean and variance (Gaussian kernel density estimation).
Embeds the estimated LCT distribution into the reward function to guide the agent toward human-like lane change durations.
Ensures multi-objective optimization of safety, efficiency, and human-like behavior.
-Experimental Results
The model achieves the following performance in test environments
Safety: Dangerous driving behaviors (TTC < 3s) reduced by 75.4%.
Efficiency: Lane-changing vehicle driving efficiency improved by 15.3%.
Reliability: 100% collision-free lane changes in all test scenarios.

## Citation
If you use this code in your research, please cite the original paper:

@article{yu202Xrlanechange,
  title={Multi-Objective Deep Reinforcement Learning-Driven Human-Like Lane-Changing Model in a Dynamic Interaction-Aware Connected Environment},
  author={[Author]},
  journal={[Journal]},
  year={202X},
  volume={[Volume]},
  number={[Number]},
  pages={[Pages]}
}
