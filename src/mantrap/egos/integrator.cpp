#include "mantrap/agents/egos/integrator.h"


mantrap::IntegratorDTEgo::IntegratorDTEgo(const double x, const double y, const double dt)
: mantrap::DTEgo<mantrap::Vector2D, mantrap::Vector2D>(mantrap::Vector2D(x, y), dt)
{
    _history.push_back(mantrap::PoseStamped2D(this->pose(), 0));
}


mantrap::IntegratorDTEgo::IntegratorDTEgo(const mantrap::Vector2D & position, const double dt)
: mantrap::DTEgo<mantrap::Vector2D, mantrap::Vector2D>(position, dt)
{
    _history.push_back(mantrap::PoseStamped2D(this->pose(), 0));
}


mantrap::Vector2D
mantrap::IntegratorDTEgo::dynamics(const mantrap::Vector2D state,
                                   const mantrap::Vector2D action) const
{
    const mantrap::Vector2D state_next(state.x + action.x * this->dt(), state.y + action.y * this->dt());
    return state_next;
}


mantrap::Vector2D
mantrap::IntegratorDTEgo::position_from_state(const mantrap::Vector2D & state) const
{
    return state;
}


mantrap::Pose2D
mantrap::IntegratorDTEgo::pose_from_state(const mantrap::Vector2D & state) const
{
    const mantrap::Pose2D pose(state.x, state.y, 0);
    return pose;
}
