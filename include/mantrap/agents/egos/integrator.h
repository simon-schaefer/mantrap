#ifndef MANTRAP_EGOS_INTEGRATOR_H
#define MANTRAP_EGOS_INTEGRATOR_H

#include "mantrap/agents/egos/abstract.h"


namespace mantrap {

class IntegratorDTEgo : public mantrap::DTEgo<mantrap::Position2D, mantrap::Velocity2D> {


public:
    IntegratorDTEgo(const mantrap::Position2D position);

    mantrap::Position2D dynamics(const mantrap::Position2D state, const mantrap::Velocity2D action) const;

};
}



#endif //MANTRAP_EGOS_INTEGRATOR_H
