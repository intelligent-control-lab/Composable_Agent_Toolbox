## Model-based Control

### High gain/saturation, NO Safe Control
Fast heading adjustment; efficient execution; jittering motion

### High gain/saturation, WITH Safe Control
Stuck around obstacle; zigzaging around $\phi=0$

### Low gain/saturation, WITH Safe Control
Go around the obstalce; efficient safe behavior

### Q: Why doesn't this happen to Ballmodel?
The safety monitor always pushes the agent away. As long as that perturbation doesn't align with the goal direction, the agent guarantees to generate a different action than before.

With unicycle model, safety is mainly guaranteed by changing the heading, which would be quickly reverted by aggressive feedback control.

---

## Model-free (RL) Control

`python flat_reach_safety_gym_RL.py --env FlatReachSafetyGym --render --demo --demo_path /home/ruic/Documents/RESEARCH/ICL/Composable_Agent_Toolbox/examples/output/FlatReachSafetyGym_ISSA_None_C400_H256_Nm100.0_lr0.0001 --demo_ep 1400`

### Expected behaviors
- Goal tracking
- Preemptive collision avoidance

### Unexpected/undesired behaviors: stalling

#### Why?
When obstacle is in the way, moving to goal normally receives low reward

#### How does RL learn to go around?
Random exploration discovers that when the goal is nearer, moving in the tangent direction helps.

#### Why does RL still stall?
The above behavior hasn't been explored fully in all states (continuous space).

#### How to further improve?
- Add more state features helping the RL agent to realize the relative relations between goal and obstacle (e.g., relative heading, distance).
- Add reward terms that encourage tangent movements. This immediate reward would be much more informative than the goal-reaching reward which only appears after many steps. (This is the problem of sparse reward, and the solution is reward shaping.)