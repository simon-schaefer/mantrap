#include <cmath>
#include <iostream>

#include "mantrap/agents/ados/dtvado.h"


mantrap::DTVAdo::DTVAdo(const double position_x, const double position_y)
{
    _position = mantrap::Vector2D(position_x, position_y);
    _velocity = mantrap::Vector2D(0, 0);
    const mantrap::PoseStamped2D pose_initial(_position.x, _position.y, 0, 0);
    _history.push_back(pose_initial);
}


mantrap::DTVAdo::DTVAdo(const double position_x,
                        const double position_y,
                        const double velocity_x,
                        const double velocity_y)
{
    _position = mantrap::Vector2D(position_x, position_y);
    _velocity = mantrap::Vector2D(velocity_x, velocity_y);
    const double theta = atan2(velocity_y, velocity_x);
    const mantrap::PoseStamped2D pose_initial(_position.x, _position.y, theta, 0);
    _history.push_back(pose_initial);
}


mantrap::DTVAdo::DTVAdo(const mantrap::Vector2D & position,
                        const mantrap::Vector2D & velocity,
                        const mantrap::Trajectory& history)
: _position(position), _velocity(velocity)
{
    _history.resize(history.size());
    _history.insert(_history.begin(), history.begin(), history.end());
    const double theta = atan2(velocity.y, velocity.y);
    const mantrap::PoseStamped2D pose_initial(_position.x, _position.y, theta, 0);
    _history.push_back(pose_initial);
}


void mantrap::DTVAdo::update(const mantrap::Vector2D& acceleration, const double dt)
{
    const mantrap::Vector2D velocity_new = acceleration * dt + _velocity;
    _position = acceleration * 0.5 * pow(dt, 2) + _velocity * dt + _position;
    _velocity = velocity_new;

    const double theta = atan2(velocity_new.y, velocity_new.y);
    const mantrap::PoseStamped2D pose_new(_position.x, _position.y, theta, dt);
    _history.push_back(pose_new);
}