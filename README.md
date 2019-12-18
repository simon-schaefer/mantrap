# MANTRAP
Minimal interferring Interactive Risk-aware Planning for multimodal and 
time-evolving obstacle behaviour

## Description
Planning safe human-robot interaction is a necessary towards the widespread 
integration of autonomous systems in the society. However, while instinctive 
to humans, socially compliant navigation is still difficult to quantify due 
to the stochasticity in people’s behaviors. Previous approaches have either 
strongly simplified the multimodal and time-varying behaviour of humans, 
applied hardly tractable methods lacking safety guarantees or were simply not 
computationally feasible. Therefore the goal of this work to develop a 
risk-aware planning methodology with special regards on minimizing the 
interaction between human and robot and taking account the actual multi-modality
and time-evolving nature of the humans behaviour, based on the Trajectron 
model (Ivanovic 19).  


## Comments
* optimization over future points 
    * determine path candidates, perturb them using interaction grad and 
      pick the path with minimal cost (e.g. interaction force caused by ego, 
      control cost using acceleration amount) 
    * create graph structure over multiple time-steps by concatenating graphs
* evaluation interface
    * probabilistic: likelihood of perturbed trajectory over initial 
      distribution
    * deterministic: L2 distance of trajectory points
* tube based on distribution of gradients for probabilistic agent
* homotopy class --> given my base path, more globally optimal ??
* sphinx documentation 

* idea -> multi-modality ? every mode is an independent (weighted) ado
* idea -> minimal interfering, why ? natural way of walking in crowded areas
(local minima, "without a lot of considerations"), intrinsically a safe way 
of interaction and taking the "full" knowledge of the model into account (not 
just the output)
* model-based approach 
* idea -> gradient, why ? notion of direction and magnitude of violation, a lot 
of behavioral prediction models are graph-based (e.g. neural network) or can be
formulated as a graph (as social forces)
 
* baseline: least force caused by ego over the full trajectory -> interaction
minimal (e.g. in comparison with minimizing travel-time, ...)
* baseline: in the same scene let human in one and robot in another experiment
take decisions, let humans guess who is who based on trajectories 
-> "natural" way of interacting with other agents, specifically pedestrians
* baseline: ORCA (don't share responsibility in eq. (6) but full velocity 
adaption by ego --> interaction-minimization)

* GMM as n independent single Gaussians --> integration using Kalman Filter 
equations with state (x, v) and no input (s. Tim)
