# Benchmark
Benchmark for model-based planning and control.

### Model

Symbolic models for linear/nonlinear systems;

### Planner

Optimization-based planners;

### Estimator

Estimation and prediction of states and parameters;

### Controller

Feedback control, feedforward control, safe control;

## Requirements

The python version is 3.8. But the code should be compatible with all 3.x version.

We specified the dependencies in `env.yml`. You can reproduce the conda environment by

```bash
conda env create --name $your_env_name -f env.yml
```

If you already have an environment and just want to make sure existing packages align with the specification. Run

```bash
conda env update --name $your_env_name -f env.yml
```

If you want to use the Mujoco env. Install mujoco_py with
```
pip install mujoco_py==2.0.2.8
```

## Usage

See [examples](https://github.com/intelligent-control-lab/Benchmark/tree/master/examples) for more information.

