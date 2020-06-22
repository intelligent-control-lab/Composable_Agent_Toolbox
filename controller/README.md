# Controller Module

This controller module contains class definition including:

Basic classes:

- Controller_Base
- Controller
- Zero_Controller
- Controller_Manager

User-defined controller classes:

- PID
- Vel_FF

## How to Use

### 1. Create a controller

```python
import controller
controller = controller.Controller(control_spec, model)
```

`control_spec` should be given in the following format:

```python
{
    'coordination': 'CCC',
    'feedback': 'PID',
    'feedforward': 'Vel_FF',
    'safety': 'SMA',
    'params': {
        'coordination': params,
        'feedback': params,
        'feedforward': params,
        'safety': params
    }
}
```

The controller key can be ignored if it's not used, it will be filled with an zero controller automatically.

E.g. a PID controller specs:

```python
{
    'feedback': 'PID',
    'params': {
        'feedback': {
            'kp': [1, 1],
            'ki': [0, 0],
            'kd': [0, 0]
        }
    }
}
```

### 2. Build the controller manager

Modify the `build_controller` function in `controller_manager.py` according to the controller structure.

E.g. a feedback only controller:

```python
def build_controller(self, dt, x, goal_x, est_params):
    feedback_output = self.feedback.control(dt, x[0], goal_x[0], est_params)
    return feedback_output
```

### 3. Get control output

```python
control = controller.control(dt, est_state, agent_goal_state, est_parameter)
```

## Data Structure

### Inputs

- `dt`: float
- `x`:
- `goal_x`:
- `model`:
- `est_params`:
- `control_spec`: dict, see 'How to use' for detailed format

### Outputs

- `control`: a column vector has the same length as `model.u_shape`
