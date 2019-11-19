#include <eigen3/Eigen/Dense>
#include "gtest/gtest.h"

#include "mantrap/agents/ados/single_mode.h"


TEST(test_single_mode_ado, vpdf) {
    mantrap::Velocity2D v_mean;
    v_mean << 1.0, 0.0;
    Eigen::Matrix2d v_covariance;
    v_covariance << 1.0, 0.0, 0.0, 1.0;

    const mantrap::SingeModeDTVAdo ado(v_mean, v_covariance);
    gmmstats::Gaussian2D vpdf = ado.vpdf_self();

    EXPECT_NEAR(vpdf.mean()[0], v_mean[0], 0.1);
    EXPECT_NEAR(vpdf.mean()[1], v_mean[1], 0.1);
    EXPECT_NEAR((vpdf.covariance() - v_covariance).norm(), 0, 0.1);
}


