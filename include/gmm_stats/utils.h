#ifndef GMM_STATS_UTILS_H
#define GMM_STATS_UTILS_H

#include <eigen3/Eigen/Dense>

Eigen::Matrix2d eye2D() {
    Eigen::Matrix2d a;
    a << 1, 0, 0, 1;
    return a;
}

Eigen::Matrix2d ones2D() {
    Eigen::Matrix2d a;
    a << 1, 1, 1, 1;
    return a;
}

Eigen::Matrix2d diagonal2D(double x, double y) {
    Eigen::Matrix2d a;
    a << x, 0, y, 0;
    return a;
}



#endif //GMM_STATS_UTILS_H
