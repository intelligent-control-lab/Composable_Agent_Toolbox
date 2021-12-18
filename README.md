# Composable Agent Toolbox (CAT)
Composable benchmark for intelligent agents. The design can support 
1. modularized agents and end-to-end agents, and 
2. model-based agents and model-free agents.

### Model
The model module contains structured differentiable functions that either represent system dynamics or control policies.
The model can be analytical (i.e., ODEs, basis functions) or numerical (i.e., neural networks).
For now, we implemented analytical models for system dyanmics, which are encoded symbolically. 
Both linear/nonlinear systems are supported.

### Planner
The planner module contains a class of functions that map observation to a sequence of future states of future inputs. A planner can either be model-based or model-free. 
For now, we implemented model-based planners, especially optimization-based planners.

### Estimator
The estimator module contains a class of functions that map loss to parameter update signals. The loss can be prediction errors. The objects being estimated can either be states or parameters (either in the dynamics or in the controller/planner). These states or parameters can either from the ego agent or other agents in the environment. When we estimate the parameters, it is viewed as a learning process. 
The algorithm for the estimation can either be first order, e.g., SGD, or second order, e.g., RLS, KF, EKF, UKF.
For now, we implemented second order estimation methods for system states.

### Controller
The controller module contains a class of functions that map observation to the next control input. The only difference between a planner and a controller lies on their outputs, where the planner outputs a sequence which contains multiple future actions, while the controller outputs one future action. A controller can either be model-based or model-free. 
For now, we implemented model-based controllers, e.g., tracking control, safe control.

## Agents
### Hierarchical Agents
An agent that runs planner and controller at different frequencies.
### End-to-End Agents
An agent that use an end-to-end black-box policy.

## Requirements

We specified the dependencies in `requirements.txt`. You can reproduce the environment by

```bash
conda create -n CAT python=3.8
conda activate CAT
cd $REPO_PATH
pip install -r requirements.txt
```

If python raises error of graphviz, try reinstalling graphviz with either of the following commands
```
pip install graphviz
conda install python-graphviz
brew install graphviz
sudo apt install graphviz
```


If you want to use the Mujoco env. Install mujoco_py with
```
pip install mujoco_py==2.0.2.8
```
