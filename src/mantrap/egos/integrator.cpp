#include "mantrap/agents/egos/integrator.h"

mantrap::IntegratorDTEgo::IntegratorDTEgo(const mantrap::Position2D & position, const double dt)
: mantrap::DTEgo<mantrap::Position2D, mantrap::Velocity2D>(position, dt)
{
    _history.push_back(mantrap::PoseStamped2D(pose(), 0));
}


mantrap::Position2D
mantrap::IntegratorDTEgo::dynamics(const mantrap::Position2D state,
                                   const mantrap::Velocity2D action) const
{
    const mantrap::Position2D state_next(state.x + action.vx * _dt, state.y + action.vy * _dt);
    return state_next;
}


mantrap::Position2D
mantrap::IntegratorDTEgo::position_from_state(const mantrap::Position2D & state) const
{
    return state;
}


mantrap::Pose2D
mantrap::IntegratorDTEgo::pose_from_state(const mantrap::Position2D & state) const
{
    const mantrap::Pose2D pose(state.x, state.y, 0);
    return pose;
}
