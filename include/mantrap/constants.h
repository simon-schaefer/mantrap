#ifndef MANTRAP_CONSTANTS_H
#define MANTRAP_CONSTANTS_H

#include <eigen3/Eigen/Dense>

namespace mantrap {
    const int thorizon_default = 20;                        // forward unrolling time horizon.

    const Eigen::Vector2d sim_x_axis_default{0, 10}; // x expansion of simulation environment.
    const Eigen::Vector2d sim_y_axis_default{0, 10}; // y expansion of simulation environment.
    const double sim_dt_default = 0.1;                      // euler forward integration time step [s].
}

#endif //MANTRAP_CONSTANTS_H
