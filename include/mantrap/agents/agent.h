#ifndef MANTRAP_AGENT_H
#define MANTRAP_AGENT_H

#include "mantrap/types.h"

namespace mantrap {

class Agent {

public:

    virtual mantrap::Position2D position() const = 0;
    virtual mantrap::Trajectory history() const = 0;

};
}


#endif //MANTRAP_AGENT_H
