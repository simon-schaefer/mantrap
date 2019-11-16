#ifndef MANTRAP_AGENT_H
#define MANTRAP_AGENT_H

#include "mantrap/types.h"

class Agent
{

    virtual mantrap::Position2D position();
    virtual mantrap::State state();
    virtual mantrap::Path history();
    virtual mantrap::Path trajectory();

}


#endif //MANTRAP_AGENT_H
