# Meetings

## 11. October (Karen)
- tree-search --> brute force (Hindsight Optimisations instead ? like in "Probabilistic Planning via Determinization in Hindsight")
- Summarise modes as Gaussian Process Model (mu, sigma) and take n most probable modes
--> weighting using number of samples in mode/probability z_i in VAE latent space according to mode_i
- Gaussian over time is not a standard distribution (model non-linear, non-markovian, ...) --> re-interpolate cost function (probability map) while enforcing smoothness constraints
- optimisation cost formulation: 
J = J0 + sum_i^modes gamma_i border_function(mode_constraint) 
with J0 containing the robots traveling cost, final cost, etc. 

- Ressources: 
-- Probablistic Robotics
-- Planning Algorithms (LaValle)
-- Decision Making under Uncertaintyl(Michael Kochendörfer --> preprint, Mail)
-- Algorithms for Optimisation (Michael Kochendörfer --> Robin)