#ifndef MANTRAP_ADOS_SINGLE_MODE_H
#define MANTRAP_ADOS_SINGLE_MODE_H

#include <eigen3/Eigen/Dense>

#include "gmm_stats/gaussian2d.h"
#include "mantrap/types.h"
#include "mantrap/agents/ados/abstract.h"

namespace mantrap {

class SingeModeDTVAdo : public mantrap::DTVAdo<gmmstats::Gaussian2D> {

    mantrap::Velocity2D _velocity_mean;
    Eigen::Matrix2d _velocity_covariance;

public:
    SingeModeDTVAdo();

    SingeModeDTVAdo(const mantrap::Position2D position,
                    const mantrap::Velocity2D velocity_mean,
                    const Eigen::Matrix2d velocity_covariance = Eigen::Matrix2d{1, 0, 0, 1},
                    const mantrap::Trajectory & history = mantrap::Trajectory());

    gmmstats::Gaussian2D vpdf(const mantrap::Trajectory& history) const;
};
}


#endif //MANTRAP_ADOS_SINGLE_MODE_H
