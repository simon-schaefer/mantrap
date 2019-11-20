#include <vector>

#include "gtest/gtest.h"

#include "mantrap/agents/ados/single_mode.h"
#include "mantrap/agents/egos/integrator.h"
#include "mantrap/simulation/environment.h"


TEST(test_environment, initialization)
{
    mantrap::Position2D position(-5, 1);
    mantrap::Velocity2D v_mean(1, 0);
    Eigen::Matrix2d v_covariance;
    v_covariance << 1.0, 0.0, 0.0, 1.0;
    const mantrap::SingeModeDTVAdo ado_1(position, v_mean, v_covariance);

    position = mantrap::Position2D(6, -3.0);
    v_mean = mantrap::Velocity2D(2.0, -1.0);
    v_covariance << 1.0, 0.0, 0.0, 1.0;
    const mantrap::SingeModeDTVAdo ado_2(position, v_mean, v_covariance);

    mantrap::Position2D state(1.0, 0.01);
    mantrap::Velocity2D action(-1.0, 10);
    const mantrap::IntegratorDTEgo ego(state);

    mantrap::Environment env;
    env.add_ego(ego);
    env.add_ado(ado_1);
    env.add_ado(ado_2);

    EXPECT_EQ(env.ados().size(), 2);

    std::vector<mantrap::SingeModeDTVAdo> ados(2);
    for(int i = 0; i < 2; ++i) {
        ados[i] = std::any_cast<mantrap::SingeModeDTVAdo>(env.ados()[i]);
    }
    EXPECT_NEAR((ado_1.vpdf_self().mean() - ados[0].vpdf_self().mean()).norm(), 0, 0.1);
    EXPECT_NEAR((ado_2.vpdf_self().mean() - ados[1].vpdf_self().mean()).norm(), 0, 0.1);
    EXPECT_NEAR((ado_1.vpdf_self().covariance() - ados[0].vpdf_self().covariance()).norm(), 0, 0.1);
    EXPECT_NEAR((ado_2.vpdf_self().covariance() - ados[1].vpdf_self().covariance()).norm(), 0, 0.1);

    std::vector< std::vector<mantrap::Trajectory> > samples = env.generate_trajectory_samples(20, 10);

    EXPECT_EQ(samples.size(), 2);
    EXPECT_EQ(samples[0].size(), 10);
    for(int k = 0; k < 2; ++k) {
        EXPECT_EQ(samples[k].size(), 10);
        for(int i = 0; i < 10; ++i) {
            EXPECT_EQ(samples[k][i].size(), 20);
        }
    }
}
