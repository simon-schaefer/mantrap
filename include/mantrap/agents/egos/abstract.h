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
    double _dt;

public:
    DTEgo(const state_t & state, const double & dt)
    : _state(state), _dt(dt) {}

    // Build the trajectory from some policy and current state, by iteratively applying the model dynamics.
    // Thereby a perfect model i.e. without uncertainty and correct is assumed.
    // @param policy: sequence of inputs to apply to the robot.
    // @return trajectory: resulting trajectory (no uncertainty in dynamics assumption !).
    mantrap::Trajectory unroll_trajectory(const std::vector<action_t> & policy) {
        mantrap::Trajectory trajectory(policy.size() + 1);

        // initial trajectory point is the current state.
        trajectory[0] = mantrap::PoseStamped2D(pose(), 0);

        // every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        state_t state_at_t = state();
        mantrap::Pose2D pose_at_t;
        for(int i = 0; i < policy.size(); ++i) {
            state_at_t = dynamics(state_at_t, policy[i]);
            pose_at_t = pose_from_state(state_at_t);
            trajectory[i + 1] = mantrap::PoseStamped2D(pose_at_t, (i + 1) * _dt);
        }
        return trajectory;
    }

    virtual state_t dynamics(const state_t state, const action_t action) const = 0;


    // Transform internal state to 2D pose (x, y, theta) and 2D position (x, y).
    virtual mantrap::Position2D position_from_state(const state_t & state) const = 0;
    virtual mantrap::Pose2D pose_from_state(const state_t & state) const = 0;


    state_t state() const                       { return _state; }
    mantrap::Pose2D pose() const                { return pose_from_state(state()); }
    mantrap::Position2D position() const        { return position_from_state(state()); }
    mantrap::Trajectory history() const         { return _history; }

};
}


#endif //MANTRAP_EGOS_ABSTRACT_H
