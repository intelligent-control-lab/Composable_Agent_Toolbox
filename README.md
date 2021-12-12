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

## Usage

See [examples](https://github.com/intelligent-control-lab/Benchmark/tree/master/examples) for more information.

## DSTA

### DONE
- General
    - Now computation agent spec is read from env yaml (as file path) instead of coding in py
- Planning Models
    - Double integrator model
- Planner
    - Naive planner (interpolation, does not use model)
    - Integrator planner (needs double/triple/etc. integrator model)
- Control Models
    - Ballmodel
    - Unicycle
        - Non linear PID
- Feedback Controller
    - Naive (use arbitrary control models)

### Assumption
- Planner takes goal + goal type defined by task
- Controller will always use full dimension of planner traj, but can ignore time deritatives
- Control model should be consistent with/same as that of world agent.
    - The world agent needs to provid all info needed by control model

## HW1 Safe Control

## HW2 Safe Planning
`python flat_reach.py`

### Note
- `examples/configs/flat_reach_agent_2.yaml` uses `CFSPlanner` defined in `agent/planner/Planner.py` which calls `_CFS()` to solve planning.
- Planner is invoked at `agent/model_based_agent.py:51`.