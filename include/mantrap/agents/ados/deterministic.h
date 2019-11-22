#ifndef MANTRAP_ADOS_DETERMINISTIC_H
#define MANTRAP_ADOS_DETERMINISTIC_H

#include <eigen3/Eigen/Dense>

#include "gmm_stats/gaussian2d.h"
#include "mantrap/constants.h"
#include "mantrap/types.h"
#include "mantrap/agents/ados/abstract.h"

namespace mantrap {

    class DeterministicDTV : public mantrap::Agent {

        mantrap::Position2D _position;
        mantrap::Velocity2D _velocity;
        mantrap::Trajectory _history;

    public:
        DeterministicDTV();

        DeterministicDTV(const mantrap::Position2D & position,
                         const mantrap::Velocity2D & velocity,
                         const mantrap::Trajectory & history = mantrap::Trajectory());

        mantrap::Position2D position() const        { return _position; }
        mantrap::Pose2D pose() const                { return mantrap::Pose2D(_position.x, _position.y, 0)}
        mantrap::Trajectory history() const         { return _history; }
    };
}


#endif //MANTRAP_ADOS_DETERMINISTIC_H
