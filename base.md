# Ideas Spreadsheet


## Path planning approaches
The path planning must deal with multimodal, stochastically time-evolving disturbances. While the position of each obstacles for t1 = t0 + 1 is a GMM (analytically determinable), for t > t1 the distribution is basically anything, since it depends on a sample of the distribution at t = t1 (nested GMM). Therefore the problem is neither linear nor markovian ... Therefore there are basically two main approaches tackling the problem: 

1) Choose action while unrolling: As described above working with the distribution for t > t1 might be hard, thus we might predict only one time-step at a time, plan one path, append the trajectories/histories according to the prediction and repredict. Repeat this loop until the planning time horizon is reached. 
+ ppdf are GMMs (not any distribution)
+ interaction-aware (i.e. takes into account the reaction to the robot's actions)
- might be greedy (hardly globally optimal) since only optimising over one step
==> Monte Carlo Tree Search


2) Predict the full time-horizon at once 
+ might be more globally optimal 
- loss of interaction awareness
==> Chance-constrained MPC 



## Chance-constrained MPC
CCMPC introduces the risk of failure as a constraint in a sense that the accumulated (uniformly weighted, linear sum) risk over the planning horizon should be smaller than some constant, while the risk taken at every time-step r_k is introduced as optimisation variable and can therefore be "smartly" distributed over the planning horizon. However Gaussian distributions (or GMMs or whatever pdf) are not in general linear or convex, therefore they cannot be used directly in a constraint, but has to be reformulated as following: 

--> Modern Portfolio Theory: w^T * sigma * w - q * R^T * w for representing unimodal Gaussians in a constraint (R = [mu_1, ..., mu_N])

--> Using Taylor series expansion to express the distance to every modes mean ||x_k - z_k||_2

--> Represent Gaussian eps-probability level set as polytopic constraint (in simplest case just linear inequality constraints e.g. x_5 < [4.0, 1.3])

--> Formulate distance to Gaussian as cone constraint (comp. [5], [6])



## Baseline
In order to show the (hopefully) advantage gained using planning based on multimodal, stochastic, time-evolving distributions or to show the trade-off between computational efficiency and optimality the obstacle distribution can be simplified in many ways:  

- Time-expanded graph search (optimal D2TS problem)
--> enforce dynamics by creating edges based on robot dynamics function
--> enforce constraints by cutting leaves which are not fulfilling these constraints
--> time-expanded graph is a static (!) graph so provable optimal solvable for discrete state and input space but very computationally complex (increases for more relaxing constraints)

- Constant-Velocity/Kalman obstacle (any problem)
--> obstacles velocity is assumed to be constant over time and equal to the initial/mean of most probable mode velocity
--> alternatively: additionally propagate uncertainty of velocity using Kalman filter (and pedestrian integrator model)
--> advantage of multimodality, time-evolving, stochasticity

- Static vpdf obstacle (any problem)
--> obstacles velocity are sampled from the same (maybe multimodal) distribution for all times
--> advantage of time-evolving

- Unimodal vpdf obstacle (any problem)
--> merely take most probable mode into account for obstacle prediction 
--> advantage of multi-modality

- RRT algorithm multiple times while selecting the path with minimum approximate path collision probability (comp. [4], Related Work)
--> efficiency ? 
--> safety guarantees ? 

- (eventually) using other pedestrian prediction models (such as Social Forces, etc.) + initial obstacle state for prediction part



## Evaluation
For evaluation some underlying distribution for the obstacles/pedestrians is assumed and sampled during testing time. The planner either knows this distribution (in order to make the evaluation  independent from perception errors) and either exploits its full knowledge (e.g. time-expanded graph search, MPC, etc.) or just part of it (e.g. single mode, initial/mean velocity) to plan a trajectory. During repeated experiments while the robot always drives the same planned trajectory, the obstacles move stochastically.  

- Monte-Carlo simulations (i.e. use derived policy and simulate N runs while sampling from distribution, comp. [3]) with the following measures
--> empirical probability of failure vs risk
--> comfort (smooth acceleration, etc.)
--> travel-time
--> minimal distance to obstacles

- (eventually) human in the loop comparison (steering wheel)



## Trajectron Information [7]
- query time with n pedestrians ==> ~ 20 ms
- roughly average or maximum number of modes per pedestrian ==> max 16
- how do we get discrete trajectories for every mode, when sampling after every time-step (not like a branching tree) ? ==> iteratively sampling n trajectories sequences by re-applying the model



[1] On Infusing Reachability-Based Safety Assurance within Probabilistic Planning Frameworks for Human-Robot Vehicle Interactions
[2] Probabilistic Planning via Determinization in Hindsight
[3] Chance-Constrained Dynamic Programming with Application to Risk-Aware Robotic Space Exploration
[4] Monte Carlo Motion Planning for Robot Trajectory Optimization Under Uncertainty
[5] On Distributionally Robust Chance-Constrained Linear Programs
[7] The Trajectron: Probabilistic Multi-Agent Trajectory Modeling With Dynamic Spatiotemporal Graphs



# Questions
- Path planning while unrolling: Which points do we add to the history of obstacles/pedestrians after predicting one step ? We do know the distribution and its a GMM but we do need to add one specific point ... 
- Path planning while unrolling: How do we perform state transitions/simulation for Monte Carlo tree search ? Do we have to re-evaluate the trajectron model every time ?



####################################################################################################
### OLD ############################################################################################
####################################################################################################

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

- Grid propagation: pyramidal grid prediction i.e. start with high-resoluted GMM grid at t=1, then insert every grid point as next point in trajectron model (like as being sampled from GMM) to build up t=2 grid, repeat until t=N with decreasing resolution (uncertainty in time increases anyway)
--> "smart" grid: only use grid points with P(x) > eps

- Kalman Filter propagation of initial gaussian distribution (single-integrator model)
--> very computational efficient
--> neglects any form of non-linear spatiotemporal evolvement 
