#ifndef MANTRAP_ADOS_ABSTRACT_H
#define MANTRAP_ADOS_ABSTRACT_H

#include <cmath>
#include <vector>

#include <eigen3/Eigen/Core>

#include "gmm_stats/gmm_stats.h"
#include "mantrap/agents/agent.h"
#include "mantrap/constants.h"
#include "mantrap/types.h"

namespace mantrap {

class DTVAdo : mantrap::Agent {

protected:

    mantrap::Vector2D _position;
    mantrap::Vector2D _velocity;
    mantrap::Trajectory _history;

public:
    DTVAdo(const double position_x, const double position_y);

    DTVAdo(const double position_x, const double position_y, const double velocity_x, const double velocity_y);

    DTVAdo(const mantrap::Vector2D & position = mantrap::Vector2D(0, 0),
           const mantrap::Vector2D & velocity = mantrap::Vector2D(0, 0),
           const mantrap::Trajectory& history = mantrap::Trajectory());

    // Update ado state (position, velocity and history) using its single integrator dynamics.
    void update(const mantrap::Vector2D& acceleration, const double dt);

    mantrap::Vector2D position() const      { return _position; }
    mantrap::Pose2D pose() const
    {
        const double theta = atan2(_velocity.y, _velocity.x);
        return mantrap::Pose2D(_position.x, _position.y, theta);
    }

    mantrap::Vector2D velocity() const      { return _velocity; }
    double speed() const                    { return _velocity.norml2(); }

    mantrap::Trajectory history() const     { return _history; }

};
}

#endif //MANTRAP_ADOS_ABSTRACT_H
