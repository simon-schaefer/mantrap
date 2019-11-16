#ifndef MANTRAP_TYPES_H
#define MANTRAP_TYPES_H

#include <eigen3/Eigen/Dense>

namespace mantrap
{
    typedef Eigen::Vector2d Position2D;
    typedef Eigen::VectorXd State;
    typedef Eigen::Matrix2Xd Path;
}

#endif //MANTRAP_TYPES_H
