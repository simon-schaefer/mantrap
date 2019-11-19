#include "mantrap/agents/egos/integrator.h"

mantrap::IntegratorDTEgo::IntegratorDTEgo(const mantrap::Position2D position)
: mantrap::DTEgo<mantrap::Position2D, mantrap::Velocity2D>(position) {}


mantrap::Position2D mantrap::IntegratorDTEgo::dynamics(const mantrap::Position2D state,
                                                       const mantrap::Velocity2D input) const {
    return state + input;
}
