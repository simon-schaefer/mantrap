#ifndef MANTRAP_ADOS_ABSTRACT_H
#define MANTRAP_ADOS_ABSTRACT_H

#include <vector>

#include <eigen3/Eigen/Core>

#include "gmm_stats/gmm_stats.h"
#include "mantrap/agents/agent.h"
#include "mantrap/constants.h"
#include "mantrap/types.h"

namespace mantrap {

class DTVAdo : mantrap::Agent {

protected:

    mantrap::Position2D _position;
    mantrap::Velocity2D _velocity;
    mantrap::Trajectory _history;

public:
    DTVAdo(const double position_x, const double position_y);

    DTVAdo(const double position_x, const double position_y, const double velocity_x, const double velocity_y);

    DTVAdo(const mantrap::Position2D & position = mantrap::Position2D(0, 0),
           const mantrap::Velocity2D & velocity = mantrap::Velocity2D(0, 0),
           const mantrap::Trajectory& history = mantrap::Trajectory());

    mantrap::Position2D position() const    { return _position; }
    mantrap::Pose2D pose() const            { return mantrap::Pose2D(_position.x, _position.y, 0); }
    mantrap::Velocity2D velocity() const    { return _velocity; }
    mantrap::Trajectory history() const     { return _history; }

};
}

#endif //MANTRAP_ADOS_ABSTRACT_H
