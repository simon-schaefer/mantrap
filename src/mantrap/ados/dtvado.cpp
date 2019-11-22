#include "mantrap/agents/ados/dtvado.h"


mantrap::DTVAdo::DTVAdo(const double position_x, const double position_y)
{
    _position = mantrap::Position2D(position_x, position_y);
}


mantrap::DTVAdo::DTVAdo(const double position_x,
                        const double position_y,
                        const double velocity_x,
                        const double velocity_y)
{
    _position = mantrap::Position2D(position_x, position_y);
    _velocity = mantrap::Velocity2D(velocity_x, velocity_y);
}


mantrap::DTVAdo::DTVAdo(const mantrap::Position2D & position,
                        const mantrap::Velocity2D & velocity,
                        const mantrap::Trajectory& history)
: _position(position), _velocity(velocity)
{
    _history.resize(history.size());
    _history.insert(_history.begin(), history.begin(), history.end());
    _history.push_back(mantrap::PoseStamped2D(position));
}
