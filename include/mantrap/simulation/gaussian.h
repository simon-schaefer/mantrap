#ifndef MANTRAP_SIM_GAUSSIAN_H
#define MANTRAP_SIM_GAUSSIAN_H

#include <vector>

#include <eigen3/Eigen/Dense>

#include "gmm_stats/gaussian2d.h"
#include "mantrap/constants.h"
#include "mantrap/simulation/abstract.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Simulation based on Single Gaussian Velocity Distribution /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mantrap {

template<typename ego_t>
class GaussianSimulation : public Simulation<ego_t, std::vector<mantrap::Trajectory>> {

    std::vector<gmmstats::Gaussian2D> _ado_velocity_distributions;

public:
    GaussianSimulation(const ego_t & ego,
                       const mantrap::Axis & xaxis = mantrap::sim_x_axis_default,
                       const mantrap::Axis & yaxis = mantrap::sim_y_axis_default,
                       const double dt = mantrap::sim_dt_default)
    : mantrap::Simulation<ego_t, std::vector<mantrap::Trajectory>>(ego, xaxis, yaxis, dt) {}

    // Generate trajectory samples for each ado in the environment.
    // Therefore iterate over all interally stored ados and sample from their assigned (single Gaussian)
    // distribution function. Since the samples are iid, i.e. do not depend on the history (previously sampled
    // or histroy state values), the sampling procedure is the same for every timestep.
    // @param thorizon: number of timesteps to predict.
    // @param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
    std::vector<std::vector<mantrap::Trajectory>> predict(
            const int thorizon = mantrap::thorizon_default,
            const mantrap::Trajectory & ego_trajectory = mantrap::Trajectory()) const
    {
        assert(thorizon > 0);
        assert(this->_ados.size() == _ado_velocity_distributions.size());

        const int num_samples = 20;
        const int num_ados = this->_ados.size();
        std::vector<std::vector<mantrap::Trajectory>> samples(num_ados);
        for(int i = 0; i < num_ados; ++i)
        {
            samples[i].resize(num_samples);

            // Sample possible (future) trajectories given the currently stored, thorizon steps in the future.
            // Therefore iteratively call the pdf() method, sample the next velocity from that pdf, forward integrate the
            // samples velocity to obtain the next position, append the history and repeat until the time horizon thorizon)
            // is reached, for each trajectory sample. Each trajectory starts with the last point of the objects history.
            std::vector<mantrap::Trajectory> ado_samples(num_samples);
            const mantrap::PoseStamped2D initial_pose(this->_ados[i].pose(), 0);
            for(int j = 0; j < num_samples; ++j)
            {
                mantrap::Trajectory sample_history = this->_ados[i].history();

                // First position in trajectory is current position.
                ado_samples[j].resize(thorizon);
                ado_samples[j][0] = initial_pose;

                // Obtain next position by forward integrating sample from internal velocity distribution, conditioned
                // on the full history (i.e. the internal history + previously sampled positions).
                mantrap::Vector2D velocity_t;
                for(int t = 1; t < thorizon; ++t)
                {
                    velocity_t.from_eigen(_ado_velocity_distributions[i].sample());
                    ado_samples[j][t].pose.x = ado_samples[j][t-1].pose.x + velocity_t.x * this->_dt;
                    ado_samples[j][t].pose.y = ado_samples[j][t-1].pose.y + velocity_t.y * this->_dt;
                    ado_samples[j][t].t = ado_samples[j][t-1].t + this->_dt;
                }
            }

            // Insert samples for ith ado to samples.
            samples[i] = ado_samples;
        }
        return samples;
    }

    // Add another ado to the simulation. Since the internal simulation model is based on Gaussian velocity
    // distribution each ado has to assigned to a Gaussian velocity distribution (mean + variance).
    void add_ado(const mantrap::DTVAdo & ado, const Eigen::Matrix2d & velocity_covariance)
    {
        const gmmstats::Gaussian2D distribution(ado.velocity().to_eigen(), velocity_covariance);
        _ado_velocity_distributions.push_back(distribution);
        Simulation<ego_t, std::vector<mantrap::Trajectory>>::add_ado(ado);
    }

};
}


#endif //MANTRAP_SIM_GAUSSIAN_H
