#ifndef MANTRAP_ADOS_ABSTRACT_H
#define MANTRAP_ADOS_ABSTRACT_H

#include <vector>

#include "mantrap/agents/agent.h"
#include "mantrap/types.h"


class DTVAdo {

protected:

    mantrap::Trajectory _history;
    float _dt;
    int _num_modes;

public:


    std::vector<mantrap::Trajectory> trajectory_samples();

    int num_modes();
    mantrap::Position2D position();
    mantrap::State state();
    mantrap::Trajectory history();

};

inline mantrap::Position2D DTVAdo::position() {
    return _history.back();
}

inline mantrap::State DTVAdo::state() {
    return position();
}

inline mantrap::Trajectory DTVAdo::history() {
    return _history;
}

inline int DTVAdo::num_modes() {
    return _num_modes;
}

#endif //MANTRAP_ADOS_ABSTRACT_H
