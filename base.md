# Ideas Spreadsheet

## Handling risk cavities (multimodality, non-markovian)
- brute-force: tree-search over possible trajectories, mapping to reward function and deciding for the trajectory with largest reward ([1]), paralleled on GPU --> using Hindsight Optimisations instead for more efficient search ([2])

- Mode reduction and modelling: summarise modes to bundles which are modelled by Gaussian Process Models (e.g. "cones"), take the n most probable modes (weighted by the probabilities z_i in the VAE's latent space creating mode_i in bundles
--> optimisation cost formulation: 
J = J0 + sum_i^modes gamma_i border_function(mode_constraint) 
with J0 containing the robots traveling cost, final cost, etc. 
--> one set of parameters over all times, but quite computationally expensive
 
- Approximate the different ppdfs (positional pdf) in each mode and time-step by 2D Gaussian by sampling N trajectories for each mode and infer Gaussian distribution from it (or one 3D-xyt-Gaussian over time --> assumes smoothness over time ??)
--> GMM2D at every time step but weight does not change (unrealistic, since once a path/mode has been chosen it is usually way more likely to be continued !)   

- Changing Trajectron model to output GMM at every time-step or change "sampling" to choose most likely mode (removing randomness)

- Grid propagation: 

## Cost function
- Gaussian over time (nested Gaussians) is not a standard distribution (model non-linear, non-markovian, ...) 
--> re-interpolate cost function (probability map) while enforcing smoothness constraints

## Simluation
- Obstacle disturbance in position or velocity space (more realistic in velocity, trajectron in velocity, but planner requires position), so how to transform some distribution from velocity in position space for any (not analytically known, since nested GMMs) distribution efficiently (not just particles) ? 
--> integrating velocity sequences (particles) directly and build distribution from the obtained positions. Another possibility would be to interpolate the pdf in the velocity space, obtain an analytic formulation and integrate it analytically. However we assume smoothness by interpolation in the velocity space, which probably is a larger assumption than the effect of integration error. 

## Baseline
- static pdf for every pedestrian
- deterministic Kalman prediction model for every pedestrian dependent on simple integrator model / social forces / ....

## Evaluation metrics
- empirical probability of failure in Monte-Carlo simulations (i.e. use derived policy and simulate 10000 runs while sampling from distributions, comp. [3])

[1] On Infusing Reachability-Based Safety Assurance within Probabilistic Planning Frameworks for Human-Robot Vehicle Interactions
[2] Probabilistic Planning via Determinization in Hindsight
[3] Chance-Constrained Dynamic Programming with Application to Risk-Aware Robotic Space Exploration

# Trajectron
- query time with n pedestrians ==> ~ 20 ms
- roughly average or maximum number of modes per pedestrian ==> max 16
- how do we get discrete trajectories for every mode, when sampling after every time-step (not like a branching tree) ? ==> iteratively sampling n trajectories sequences by re-applying the model


# Definitions
- Time Consistency of Risk Measure: A dynamic risk measure {ρk,N }Nk=0 is called time-consistent if, for all 0 ≤ l < k ≤ N and all sequences Z, W ∈ Zl,N, the conditions 
Zi =Wi, i=l,···,k−1, and ρk,N(Zk,··· ,ZN) ≤ ρk,N(Wk,··· ,WN),
imply that ρl,N(Zl,··· ,ZN) ≤ ρl,N(Wl,··· ,WN).


# Questions
- Baseline ? (Probabilistic graph search, just like Astar with time-dependent weights existing ?)
- most probable path (so what is the maximum of the overall GMM distribution) ? 