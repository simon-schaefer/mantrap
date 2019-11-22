#ifndef MANTRAP_SIM_GAUSSIAN_H
#define MANTRAP_SIM_GAUSSIAN_H

#include <vector>

#include <eigen3/Eigen/Dense>

#include "gmm_stats/gaussian2d.h"
#include "mantrap/constants.h"
#include "mantrap/simulation/abstract.h"

namespace mantrap {

template<typename ego_t>
class GaussianSimulation : public Simulation<ego_t, std::vector<mantrap::Trajectory>> {

    std::vector<gmmstats::Gaussian2D> _ado_velocity_distributions;

public:
    GaussianSimulation(const ego_t & ego,
                       const mantrap::Axis & xaxis = mantrap::sim_x_axis_default,
                       const mantrap::Axis & yaxis = mantrap::sim_y_axis_default,
                       const double dt = mantrap::sim_dt_default);

    // Generate trajectory samples for each ado in the environment.
    // Therefore iterate over all internally stored ados and call the generate sampling function. The number of
    // samples thereby describes the number of generated samples per ados, i.e. the total size of the (nested)
    // returned vector is num_ados * num_samples, while each trajectory has thorizon length.
    // @param thorizon: length of trajectory samples, i.e. number of predicted time-steps.
    // @param num_samples: number of trajectory samples per ado.
    // @return vector of sampled future trajectories (num_samples -> thorizon, 2).
    std::vector<std::vector<mantrap::Trajectory>> predict(
            const int thorizon = mantrap::thorizon_default,
            const mantrap::Trajectory & ego_trajectory = mantrap::Trajectory()) const;

    // Add another ado to the simulation. Since the internal simulation model is based on Gaussian velocity
    // distribution each ado has to assigned to a Gaussian velocity distribution (mean + variance).
    void add_ado(const mantrap::DTVAdo & ado,
                 const mantrap::Velocity2D & velocity_mean,
                 const Eigen::Matrix2d & velocity_covariance);

};
}


#endif //MANTRAP_SIM_GAUSSIAN_H
