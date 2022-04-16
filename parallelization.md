# Composable Agent Toolbox (CAT) Parallelization
The objective of this subproject is to parallelize agent computation from environment simulation, allowing for more flexible and realistic benchmarking. Ideally, the user can simply choose between parallel vs. sequential simulation at runtime, with this choice being independent of other user configurations.

## Multiprocessing
One way of achieving parallelized computation is through Python's ```multiprocessing``` library. As a proof of concept, a multiprocessed version of ```flat_evade.py``` is in development. For simplicity, only a single agent is currently being used. Files with multiprocessing structure are denoted with ```_mp``` at the end of the file name (e.g. ```flat_evade_mp.py```), and multiprocessing classes are denoted with ```MP``` at the end of the class name (e.g. ```FlatEvadeEnvMP```).

### Main Process (```examples/flat_evade_mp.py```)
- Initialize a server process manager object (```multiprocessing.Manager()```), and using this initialize shared object proxies for actions, sensor data, and simulation record (```Manager.dict()```, ```Manager.Queue()```).
- Initialize a lock object (```Manager.Lock()```) to prevent race conditions among processes when accessing shared memory.
- Initialize agent process(es) and env process (```multiprocessing.Process()```), passing object proxies, lock, and some constant for the number of simulation iterations.
- Start and join processes, then pop values from record proxy for evaluation.

### Agent Process (```agent/model_based_agent_mp.py/ModelBasedAgentMP::action()```)
- For specified number of iterations:
  - Get latest sensor data from shared proxy using lock.
  - Compute agent action using sensor data.
  - Update actions shared proxy with latest action using lock.

### Env Process (```env/flat_evade_env_mp.py/FlatEvadeEnvMP::step()```)
- For specified number of iterations:
  - Get latest actions from shared proxy using lock.
  - Simulate env using actions.
  - Update sensor data shared proxy with latest sensor data using lock.

### Configuration (```examples/configs/flat_evade_agent_1_mp.yaml```)
- Largely the same as configuration in ```flat_evade_agent_1.yaml```, but with some key differences:
  - The model/planning/spec ```dT``` parameter has been removed. It is now calculated within the agent process itself as the time between consecutive computations.
  - The global ```cycle_time``` parameter has been added. It controls the frequency at which the agent process computes/updates actions.

### Usage
```bash
cd $REPO_PATH
cd examples
python flat_evade_mp.py
```
