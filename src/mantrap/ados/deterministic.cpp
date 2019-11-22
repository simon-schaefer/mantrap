#include "mantrap/agents/ados/deterministic.h"


mantrap::DeterministicDTV::DeterministicDTV()
{
    _position = mantrap::Position2D(0, 0);
    _velocity = mantrap::Velocity2D(0, 0);
    _history.push_back(mantrap::PoseStamped2D(_position));
};


mantrap::DeterministicDTV::DeterministicDTV(const mantrap::Position2D & position,
                                            const mantrap::Velocity2D & velocity,
                                            const mantrap::Trajectory & history)
: _position(position), _velocity(velocity)
{
    _history.resize(history.size());
    _history.insert(_history.begin(), history.begin(), history.end());
    _history.push_back(mantrap::PoseStamped2D(position));
}
