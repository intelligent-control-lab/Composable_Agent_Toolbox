# Composable Agent Toolbox (CAT) Parallelization
The objective of this subproject is to parallelize agent computation from environment simulation, allowing for more flexible and realistic benchmarking. Ideally, the user can simply choose between parallel vs. sequential simulation at runtime, with this choice being independent of other user configurations.

## Multiprocessing
### Main Script (Main Process)
```examples/sim_mp.py```

Enables user to run any configured simulation in parallel using multiprocessing.
- Read user configurations from ```config.yaml```.
- Initialize a server process manager object, and use this to initialize shared object proxies for actions, sensor data, and simulation record.
- Initialize, start and join env and agent processes, passing object proxies, lock, etc.
- Pop values from record proxy and evaluate.

### Agent Multiprocessing Wrapper (Agent Process)
```agent/mp_wrapper.py```

Wrapper class enabling seamless simulation of any agent object within its own process. Wrapper object maintains all functionality of agent object. 

Ex: ```agent = agent.ModelBasedAgent(agent_spec)``` -> ```agent = agent.MPWrapper(agent.ModelBasedAgent(agent_spec))```

```agent/mp_wrapper.py::action_loop()```

Use in place of conventional ```action()``` to pass as process target method.
- While env running:
  - Get latest sensor data from shared proxy.
  - Send sensor data to agent object to compute action.
  - Update actions shared proxy with latest action.

### Env Multiprocessing Wrapper (Env Process)
```env/mp_wrapper.py```

Wrapper class enabling seamless simulation of any env object within its own process. Wrapper object maintains all functionality of env object.

Ex: ```env = env.FlatEvadeEnv(env_spec, agents)``` -> ```env = env.MPWrapper(env.FlatEvadeEnv(env_spec, agents))```

```env/mp_wrapper.py::step_loop()```

Use in place of conventional ```step()``` to pass as process target method.
- For specified number of iterations:
  - Get latest actions from shared proxy using lock.
  - Simulate env using actions.
  - Update sensor data shared proxy with latest sensor data using lock.

### Configuration 
```examples/config.yaml```

Central configuration file for both sequential and parallel simulation. 
- Specifies types and config filenames for ```env``` and ```agents```.
- Specifies number of ```iters``` to simulate and whether to ```render``` simulation.
- Specifies ```debug``` configurations (```render_traj```, ```log_agent_state```, etc.).

**Note:** All agent config files must contain a global ```cycle_time``` parameter to be compatible with multiprocessing. This parameter specifies the computation frequency of the agent's process.

### Usage
```bash
cd $REPO_PATH
cd examples
python sim_mp.py # parallel simulation
python sim_seq.py # sequential simulation
```
