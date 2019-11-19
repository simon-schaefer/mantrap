#ifndef MANTRAP_EGOS_ABSTRACT_H
#define MANTRAP_EGOS_ABSTRACT_H

#include <vector>

#include "mantrap/agents/agent.h"

namespace mantrap {

template<typename state_t, typename action_t>
class DTEgo : public mantrap::Agent {

protected:

    mantrap::Trajectory _history;
    state_t _state;

public:
    DTEgo(const state_t state)
    : _state(state) {
        _history.resize(1);
        _history[0] = mantrap::Position2D();    // position_from_state(state());
    }

    // Build the trajectory from some policy and current state, by iteratively applying the model dynamics.
    // Thereby a perfect model i.e. without uncertainty and correct is assumed.
    // @param policy: sequence of inputs to apply to the robot.
    // @return trajectory: resulting trajectory (no uncertainty in dynamics assumption !).
    mantrap::Trajectory unroll_trajectory(const std::vector<action_t> & policy) {
        mantrap::Trajectory trajectory(policy.size() + 1);

        // initial trajectory point is the current state.
        trajectory[0] = position();

        // every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        state_t state_at_t = state();
        for(int i = 0; i < policy.size(); ++i) {
            state_at_t = dynamics(state_at_t, policy[i]);
            trajectory[i + 1] = position_from_state(state_at_t);
        }
        return trajectory;
    }

    virtual state_t dynamics(const state_t state, const action_t action) const = 0;

    // Transform internal state to 2D position. Per default is the x-position the first and y-position the second
    // entry of the state vector. If not the method has to be reimplemented by the derived class.
    virtual mantrap::Position2D position_from_state(const state_t state) const {
        mantrap::Position2D position;
        position << state(0), state(1);
        return position;
    }

    state_t state() const                   { return _state; }
    mantrap::Position2D position() const    { return position_from_state(state()); }
    mantrap::Trajectory history() const     { return _history; }

};
}


#endif //MANTRAP_EGOS_ABSTRACT_H
