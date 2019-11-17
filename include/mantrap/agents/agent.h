#ifndef MANTRAP_AGENT_H
#define MANTRAP_AGENT_H

#include "mantrap/types.h"

class Agent {

public:

    virtual mantrap::Position2D position();
    virtual mantrap::State state();
    virtual mantrap::Trajectory history();
    virtual mantrap::Trajectory trajectory();

};


#endif //MANTRAP_AGENT_H
