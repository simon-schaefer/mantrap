#include <eigen3/Eigen/Dense>
#include "gtest/gtest.h"

#include "mantrap/agents/ados/single_mode.h"
#include "mantrap/agents/egos/integrator.h"


TEST(test_single_mode_ado, vpdf)
{
    mantrap::Position2D position(-1, 5);
    mantrap::Velocity2D v_mean(1, 0);
    Eigen::Matrix2d v_covariance;
    v_covariance << 1.0, 0.0, 0.0, 1.0;

    const mantrap::SingeModeDTVAdo ado(position, v_mean, v_covariance);

    EXPECT_NEAR(ado.position().x, position.x, 0.001);
    EXPECT_NEAR(ado.position().y, position.y, 0.001);

    gmmstats::Gaussian2D vpdf = ado.vpdf_self();

    EXPECT_NEAR(vpdf.mean()(0), v_mean.vx, 0.1);
    EXPECT_NEAR(vpdf.mean()(1), v_mean.vy, 0.1);
    EXPECT_NEAR((vpdf.covariance() - v_covariance).norm(), 0, 0.1);
}


TEST(test_integrator, dynamics)
{
    const mantrap::Position2D state(1.0, 0.01);
    const mantrap::Velocity2D action(-1.0, 10);
    const mantrap::IntegratorDTEgo ego(state, 1.0);

    const mantrap::Position2D state_next = ego.dynamics(state, action);

    EXPECT_NEAR(state_next.x, 0.0, 0.001);
    EXPECT_NEAR(state_next.y, 10.01, 0.001);
}
