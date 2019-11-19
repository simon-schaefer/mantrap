#include "gmm_stats/gaussian2d.h"
#include "mantrap/agents/ados/single_mode.h"


mantrap::SingeModeDTVAdo::SingeModeDTVAdo(const mantrap::Velocity2D velocity_mean,
                                          const Eigen::Matrix2d velocity_covariance,
                                          const mantrap::Trajectory & history)
: DTVAdo<gmmstats::Gaussian2D>(history, 1),
  _velocity_mean(velocity_mean),
  _velocity_covariance(velocity_covariance) {}


gmmstats::Gaussian2D mantrap::SingeModeDTVAdo::vpdf(const mantrap::Trajectory& history) const {
    return gmmstats::Gaussian2D(_velocity_mean, _velocity_covariance);
}
