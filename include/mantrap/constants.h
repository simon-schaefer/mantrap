#ifndef MANTRAP_CONSTANTS_H
#define MANTRAP_CONSTANTS_H

#include "mantrap/types.h"

namespace mantrap {
    const int thorizon_default = 20;                                // forward unrolling time horizon.

    const mantrap::Axis sim_x_axis_default(-10, 10);    // x expansion of simulation environment.
    const mantrap::Axis sim_y_axis_default(-10, 10);    // y expansion of simulation environment.
    const double sim_dt_default = 0.1;                              // euler forward integration time step [s].
}

#endif //MANTRAP_CONSTANTS_H
