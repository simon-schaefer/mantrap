# Ideas Spreadsheet

## Handling multimodality and time-variance 
- ghosts: simulate every mode as individual pedestrian which interacts independently from its other modes (cost contains weights)
- electrostatic interaction: handle position probability distribution as electrostatic field (==> parameter intensive)

## Optimization method
- Chance-constrained Dynamic Programming (comp. Chance-Constrained Dynamic Programming with Application to Risk-Aware Robotic Space Exploration)
- Dynamic Programming with ghosts --> problem: markovian independence property (w_k iid w_l, l < k) not satisfied 
- Graph search for time-varying edge cost (cost = probability to hit) --> globally optimal  solution ? 

## State Representation
- position of robot
- position of pedestrians with probability distribution as dynamics

## Baseline
- static pdf for every pedestrian

## Starting scenario
- identically distributed, time-invariant 2-modal gaussian distribution relative to starting position

## Evaluation metrics
- empirical probability of failure in Monte-Carlo simulations (i.e. use derived policy and simulate 10000 runs while sampling from distributions, comp. Chance-Constrained Dynamic Programming with Application to Risk-Aware Robotic Space Exploration)