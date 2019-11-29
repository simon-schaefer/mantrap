#include <vector>

#include "gtest/gtest.h"

#include "mantrap/agents/ados/dtvado.h"
#include "mantrap/agents/egos/integrator.h"
#include "mantrap/simulation/social_forces.h"


TEST(test_sim_social, initialization)
{
    const mantrap::DTVAdo ado(-5, -5, 3, 2);
    const mantrap::IntegratorDTEgo ego(0, 0);

    mantrap::SocialForcesSimulation sim(ego);
    const mantrap::Vector2D ado_goal(5, 5);
    sim.add_ado(ado, ado_goal);
}


TEST(test_sim_social, prediction)
{
    const mantrap::DTVAdo ado(-5, -5, 0.5, 1.0);
    const mantrap::IntegratorDTEgo ego(0, 0);

    mantrap::SocialForcesSimulation sim(ego);
    const mantrap::Vector2D ado_goal(5, 5);
    sim.add_ado(ado, ado_goal);

    std::vector<mantrap::Trajectory> ado_trajectory = sim.predict();
}