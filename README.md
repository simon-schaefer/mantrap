# MURSECO
Multimodal risk sensitive stochastic control


## Installation
```
source ops/setup.bash
```

## Problem Formulation

### Risk 
- multimodal / any distribution 
--> GMM with (N = 16) Gaussians which evolves over time (nested Gaussians != Gaussian, basically any distribution)

- not markovian 
--> distribution(t+1) depends on previous distributions(t0,...,t)


