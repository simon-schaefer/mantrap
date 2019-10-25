# Ideas Spreadsheet


## Policy planning approaches
The trajectory planning must deal with multimodal, stochastically time-evolving disturbances. While the position of each obstacles for t1 = t0 + 1 is a GMM (analytically determinable), for t > t1 the distribution is basically anything, since it depends on a sample of the distribution at t = t1 (nested GMM). Therefore the problem is neither linear nor markovian ... 

State of the art policy planning for MDP/POMDP, especially when dealing with large state-spaces, is (Deep)RL, ADP, etc. which basically estimate the cost-to-go of some state-action-pair. However as stated before, the problem is non-markovian so that these approaches would only work, if we reformulate it in a markovian way by extending the state by the full history of each object. As shown at the example of imitation learning with similar inputs this will hardly work for real environment problems, especially for safety critical applications (comp. Waymo [14] and conservation with Ed) ...

Also we should decide whether to go for trajectory or policy optimisation. While trajectory optimisation is definitely less computationally expensive, which is here very dependent on how the state is defined, as Ed (paper RSS ??) has shows iterative replanning is less optimal (and probably less stable ?) compared to finding a policy. Otherwise policy optimisation e.g. using some learning-based can be done offline and therefore be computationally very cheap online. However in the following I present approaches for both, trajectory and policy optimisation: 


##### Learning-based reduction of probability distribution to equivalent deterministic grid and apply deterministic (MPC) planning in that grid. 
* Why ? Well most of the approaches for PP basically either draw a lot of samples or try to convert the probabilistic to a deterministic problem (except CCMPC ??) which either way are based on iterative solving, but anyways are reducing information. So since if we kill information anyway, why shouldn't we do it in one shot ! 
* How should the architecture look like ? Basically it should be fully convolutional (like an image to image transformation) mapping a tppdf (or ppdf for the start) and the risk constraint to a deterministic map
* Why is that better than learning the policy/cost-to-go directly ? We are basically doing that just stepwise and therefore easier testable and it maybe allows us to guarantee certain similarities between the tppdf and the resulting deterministic map, e.g. using KL-divergence. So for static obstacles the output should be a level-set of the ppdf (dependent on the input threshold), for dynamic obstacles maybe "include" some notion of moving direction, i.e. the deterministic obstacle should be "biased" to the moving direction so that the robot "prefers" to move counter-directional to the obstacles moving direction.
* How do we train it ? In a self-supervised manner, using for example the output map of the Monte Carlo planning approach [4]
* Note: Non-trained reformulation to deterministic search problem [9]
* Pros: probably computationally efficient, general approach, no (human-made) heuristics
* Cons: hard to give guarantees on optimality or safety


##### Local optimization around trajectory that the Trajectron model predicts for the robot (!) node while the robot is modelled as a pedestrian. 
* Why ? Goal of the project is to control the robot so that it moves similarly to a pedestrian. When the robot is introduced as pedestrian node in the trajectron model, it outputs the exact trajectory "candidates" a pedestrian would take. 
* How ? Similarly to Amine's approach in path planning (comp. [12], A* -> path smoothing given dynamics -> velocity profile optimization). Instead of using A* as starting point one the prediction of the Trajectron model given its surrounding nodes is broken down to a trajectory e.g. the most probable trajectory, by MC sampling and selecting the trajectory with smallest cost that fulfils some safety constraint, etc. and used as a starting point. Thereby an interaction similarly to human beings as well as some safety constraints can be guaranteed, assuming some accuracy of the Trajectron model. Given the base trajectory an area can be defined in which the constraints still hold so that the trajectory can be optimised within the area, taking into account the dynamics of the robot as well as efficiency constraints (we assume here that the robot's dynamics are not too different to the pedestrian integrator model, so that the base trajectory is feasible, but I think this is a valid assumption at slow speed and using "small" robots). 
* Pros: intrinsically encounters interactions with other actors in the scene, baseline comes more or less for free
* Cons: probably hard to give theoretical guarantees based on the Trajectron, very high demands on the Trajectron model, overfitting to pedestrian/Trajectron scenario (non-general)


##### Chance-constrained MPC (SQP)
* Why ? When posed correctly CCMPC comes with guarantees such as safety or (first-order, iteratively local) optimality (comp. [17]) and allows "smart" risk allocation. 
* How ? CCMPC introduces the risk of failure as a constraint in a sense that the accumulated risk over the planning horizon should be smaller than some constant, while the risk taken at every time-step r_k is introduced as optimisation variable and can therefore be "smartly" distributed over the planning horizon. However Gaussian distributions (or GMMs or whatever pdf) are not in general linear or convex, therefore they cannot be used directly in a constraint, but has to be reformulated as following (re-approximate in every step, SQP). To reformulate these constraints the pdf has to be remodelled in "analytically" expressible distributions so that either Modern-Portfolio-Theory/Mean-variance-Analysis (w^T * sigma * w - q * R^T * w for representing unimodal Gaussians in a constraint with R = [mu_1, ..., mu_N], q >= 0 = risk tolerance factor, w = normalised weights [8]) can be used or e.g. just a 1st order Taylor series expansion can be used (for every modes mean ||x_k - z_k||_2 when remodelled as GMM) or as cone constraint (comp. [5], [17], Thomas, when modelled as GMM). Otherwise the pdf's eps-probability level set might be represented using polytopic constraints (in simplest case just linear inequality constraints e.g. x_5 < [4.0, 1.3]). 
* Pros: existing framework, computationally efficient (when convexified), more "globally" optimal and guaranteed locally optimal
* Cons: a lot of approximations to make it computational feasible, loss of interaction awareness since the full horizon is predicted at once



##### Sparsify problem and apply reactive collision avoidance (ORCA [13], Max Schröder's algorithm ??)
* Why ? POMDP are intractable, especially for large state and action spaces. So the question is what are useful abstractions/heuristic to make it tractable and computationally feasible. Regarding pedestrians we observe that in most of the times it is fairly accurate to treat them as "velocity vectors" only, without multimodality or probability distribution (Ed). 
* How ? Approximate pedestrians as "velocity obstacles" given their current velocity or (a little more complex) determine possible change points in their future trajectories and assume a constant, deterministic behaviour in between (comp. change point analysis [15]), take only most probable trajectory (mean in most probable mode) -> policy planning using intermediate goals
* Pros: computationally efficient, easy, easy to debug
* Cons: no optimality (at least not global, also local minimum caveats like u-shap for short planning horizon), overfitting to pedestrian problem (non-general)



##### Apply state-of-the-art POMDP real-time-policy planning approach like DESPOT ([16]) or Monte Carlo Tree Search (+ Hindsight Optimization [2])
* Pros: might work out of the box 
* Cons: might not be scalable to large action-space, little (theoretical) contributions
==> Baseline ?? 



##### Choose action while unrolling: As described above working with the distribution for t > t1 might be hard, thus we might predict only one time-step at a time, plan one path, append the trajectories/histories according to the prediction and repredict. Repeat this loop until the planning time horizon is reached. 
+ ppdf are GMMs (not any distribution)
+ interaction-aware (i.e. takes into account the reaction to the robot's actions)
- might be greedy (hardly globally optimal) since only optimising over one step



## Alternative (denied) policy planning approaches

- Monte-Carlo Planning [4] ? Plan some path given some eps-level set of the tppdf as static obstacles, if resulting trajectory does not fulfil constraints then change eps and repeat. However introducing "rigid" obstacles might not lead to "smart" risk allocation, i.e. approaches some obstacle closer in order to be more globally efficient.



## Approximating the actual PPDF
Over a time horizon t > 1 the Trajectron model predicts by sampling from the previously generated distribution. Therefore the distribution of velocities over multiple time-steps is a nested GMM, well basically anything and not really analytically expressible. While samples from the distribution are accessible by iteratively applying the model and can be easily forward integrated (since position-based information are more straight forward to integrate in most planners) getting the distributions themselves is not straight-forward. However it can be approximated: 

##### Sample trajectories from the model, integrate them and approximate the obtained distribution 
* GMMs basically are proxies for modelling some distribution, therefore for every time-step also a GMM could be used to model the obtained samples to a distribution (how many kernels ? = modes * obstacles or less)
* Each mode (i.e. z_i in latent space of Trajectron) can be sampled from individually and modelled per time-step as 2D Gaussian (sigma_xy = 0 for comp. reasons), which is requires more sample-inefficient but directly gives you a sense of importance of each Gaussians due to the value of the mode in the latent space
* Pros: fairly easy and tractable
* Cons: approximative, no regularity constraints between subsequent time-steps (however I am not sure whether this is desirable ?)


##### Adapting Trajectron model 
* Problematic about the way the Trajectron model builds vpdfs is not that they are nested but that the nesting is random and therefore not simply tractable. Therefore an approach might be to only propagate a specific point (e.g. the mean of the most probable mode) to the next LSTM cell
* Also the Trajectron might be retrained to base the next time-step on the previous prediction but predict it without sampling from the previous distribution. 




## Baseline
In order to show the (hopefully) advantage gained using planning based on multimodal, stochastic, time-evolving distributions or to show the trade-off between computational efficiency and optimality the obstacle distribution can be simplified in many ways:  

- Time-expanded graph search (optimal D2TS problem)
--> enforce dynamics by creating edges based on robot dynamics function
--> enforce constraints by cutting leaves which are not fulfilling these constraints
--> time-expanded graph is a static (!) graph so provable optimal solvable for discrete state and input space but very computationally complex (increases for more relaxing constraints)

- Constant-Velocity/Kalman obstacle (any problem) [10]
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
[8] Model-based predictive control in mean-variance portfolio optimization
[9] Risk-Sensitive Planning with Probabilistic Decision Graphs
[10] Stochastic Predictive Control of Autonomous Vehicles in Uncertain Environments
[11] Learning-based Model Predictive Control for Safe Exploration and Reinforcement Learning
[12] Map-Predictive Motion Planning in Unknown Environments
[13] Optimal Reciprocal Collision Avoidance (ORCA)
[14] ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst
[15] Continuous Meta Learning without Tasks
[16] DESPOT: Online POMDP Planning with Regularization
[17] Theoretically Guaranteed Chance-Constrained Sequential Convex Programming for Robust Trajectory Optimization


# Questions
- Path planning while unrolling: Which points do we add to the history of obstacles/pedestrians after predicting one step ? We do know the distribution and its a GMM but we do need to add one specific point ... 
- Path planning while unrolling: How do we perform state transitions/simulation for Monte Carlo tree search ? Do we have to re-evaluate the trajectron model every time ?
- In evaluation it would be nice to create random scenarios, but not all of the scenarios are in fact solvable. Therefore is there a way to check whether a given problem is well posed w/o brute force tree search ? 
