#ifndef MANTRAP_TYPES_H
#define MANTRAP_TYPES_H

#include <vector>

#include <eigen3/Eigen/Dense>

namespace mantrap {
    typedef Eigen::Vector2d Position2D;
    typedef Eigen::Vector2d Velocity2D;
    typedef std::vector<Position2D> Trajectory;
}

#endif //MANTRAP_TYPES_H
