#ifndef MANTRAP_EGOS_INTEGRATOR_H
#define MANTRAP_EGOS_INTEGRATOR_H

#include "mantrap/constants.h"
#include "mantrap/agents/egos/abstract.h"


namespace mantrap {

class IntegratorDTEgo : public mantrap::DTEgo<mantrap::Vector2D, mantrap::Vector2D> {


public:
    IntegratorDTEgo(const double x, const double y, const double dt = mantrap::sim_dt_default);

    IntegratorDTEgo(const mantrap::Vector2D & position,
                    const double dt = mantrap::sim_dt_default);

    mantrap::Vector2D dynamics(const mantrap::Vector2D state, const mantrap::Vector2D action) const;
    mantrap::Vector2D position_from_state(const mantrap::Vector2D & state) const;
    mantrap::Pose2D pose_from_state(const mantrap::Vector2D & state) const;

};
}



#endif //MANTRAP_EGOS_INTEGRATOR_H
