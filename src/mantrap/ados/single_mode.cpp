#include "gmm_stats/gaussian2d.h"
#include "mantrap/constants.h"
#include "mantrap/agents/ados/single_mode.h"

mantrap::SingeModeDTVAdo::SingeModeDTVAdo()
: DTVAdo<gmmstats::Gaussian2D>(mantrap::Position2D{0, 0},
                               mantrap::Trajectory(),
                               1,
                               mantrap::sim_dt_default)
{
    _velocity_mean = mantrap::Velocity2D(0, 0);
    _velocity_covariance << 1, 0, 0, 1;
};


mantrap::SingeModeDTVAdo::SingeModeDTVAdo(const mantrap::Position2D & position,
                                          const mantrap::Velocity2D & velocity_mean,
                                          const Eigen::Matrix2d & velocity_covariance,
                                          const double dt,
                                          const mantrap::Trajectory & history)
: DTVAdo<gmmstats::Gaussian2D>(position, history, 1, dt),
  _velocity_mean(velocity_mean),
  _velocity_covariance(velocity_covariance) {}


gmmstats::Gaussian2D
mantrap::SingeModeDTVAdo::vpdf(const mantrap::Trajectory& history) const
{
    return gmmstats::Gaussian2D(_velocity_mean.to_eigen(), _velocity_covariance);
}
