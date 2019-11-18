#ifndef GMM_STATS_TYPES_H
#define GMM_STATS_TYPES_H

#include <vector>

#include <eigen3/Eigen/Dense>

namespace GMMStats {
    typedef std::vector<Eigen::Vector2d> Matrix2Xd;
    typedef std::vector<Eigen::Matrix2d> Matrix22Xd;
}

#endif //GMM_STATS_TYPES_H
