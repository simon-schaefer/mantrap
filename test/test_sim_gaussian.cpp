#include "gtest/gtest.h"

#include "mantrap/agents/ados/dtvado.h"
#include "mantrap/agents/egos/integrator.h"
#include "mantrap/simulation/gaussian.h"


TEST(test_sim_gaussian, initialization)
{
    const mantrap::DTVAdo ado(5, -1, 3, 2);
    const mantrap::IntegratorDTEgo ego(0, 0);

    mantrap::GaussianSimulation sim(ego);
    Eigen::Matrix2d ado_cov;
    ado_cov << 1, 0, 0, 1;
    sim.add_ado(ado, ado_cov);

    EXPECT_EQ(sim.ados().size(), 1);
}