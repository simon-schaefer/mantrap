#include "mantrap/simulation/gaussian.h"

template <typename ego_t>
mantrap::GaussianSimulation<ego_t>::GaussianSimulation(
        const ego_t & ego,
        const mantrap::Axis & xaxis,
        const mantrap::Axis & yaxis,
        const double dt)
: mantrap::Simulation<ego_t, std::vector<mantrap::Trajectory>>(ego, xaxis, yaxis, dt) {}


template <typename ego_t>
std::vector<std::vector<mantrap::Trajectory>>
mantrap::GaussianSimulation<ego_t>::predict(const int thorizon, const mantrap::Trajectory & ego_trajectory) const
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
            mantrap::Trajectory sample_history = this->ados[i].history();

            // First position in trajectory is current position.
            ado_samples[j].resize(thorizon);
            ado_samples[j][0] = initial_pose;

            // Obtain next position by forward integrating sample from internal velocity distribution, conditioned
            // on the full history (i.e. the internal history + previously sampled positions).
            mantrap::Velocity2D velocity_t;
            for(int t = 1; t < thorizon; ++t)
            {
                velocity_t = _ado_velocity_distributions[i].sample();
                ado_samples[j][t].pose.x = ado_samples[j][t-1].pose.x + velocity_t.vx * this->_dt;
                ado_samples[j][t].pose.y = ado_samples[j][t-1].pose.y + velocity_t.vy * this->_dt;
                ado_samples[j][t].t = ado_samples[j][t-1].t + this->_dt;
            }
        }

        // Insert samples for ith ado to samples.
        samples[i] = ado_samples;
    }
    return samples;
}


template <typename ego_t>
void
mantrap::GaussianSimulation<ego_t>::add_ado(
        const mantrap::DTVAdo & ado,
        const mantrap::Velocity2D & velocity_mean,
        const Eigen::Matrix2d & velocity_covariance)
{
    const gmmstats::Gaussian2D distribution(velocity_mean.to_eigen(), velocity_covariance);
    _ado_velocity_distributions.push_back(distribution);
    this->add_ado(ado);
}
