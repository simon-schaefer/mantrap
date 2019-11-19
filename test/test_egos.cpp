#include "gtest/gtest.h"

#include "mantrap/agents/egos/integrator.h"


TEST(test_integrator, dynamics) {
    mantrap::Position2D state;
    state << 1.0, 0.01;
    mantrap::Velocity2D action;
    action << -1.0, 10;

    const mantrap::IntegratorDTEgo ego(state);
    const mantrap::Position2D state_next = ego.dynamics(state, action);

    EXPECT_NEAR(state_next(0), 0.0, 0.001);
    EXPECT_NEAR(state_next(1), 10.01, 0.001);
}
