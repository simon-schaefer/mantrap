#include <cmath>
#include <cstddef>
#include <vector>

#include <iostream>

#include <RVO.h>

#ifndef M_PI
const float M_PI = 3.14159265358979323846f;
#endif

#ifndef MAX_SPEED
const float MAX_SPEED = 4.0f;
#endif

#ifndef SAFE_DT
const float SAFE_DT = 10.0f;
#endif

#ifndef AGENT_RADIUS
const float AGENT_RADIUS = 1.0f;
#endif


/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;

void setupScenarioOneAgent(RVO::RVOSimulator *sim)
{
	/* Specify the global time step of the simulation. */
	sim->setTimeStep(0.25f);

	/* Specify the default parameters for agents that are subsequently added. */
	sim->setAgentDefaults(99999.9f, 100, SAFE_DT, 99999.9f, AGENT_RADIUS, MAX_SPEED);

	sim->addAgent(RVO::Vector2(0.0f, 0.0f));
	goals.push_back(RVO::Vector2(4.0f, 4.0f));
}

void setupScenarioTwoAgents(RVO::RVOSimulator *sim)
{
	/* Specify the global time step of the simulation. */
	sim->setTimeStep(0.25f);

	/* Specify the default parameters for agents that are subsequently added. */
	sim->setAgentDefaults(99999.9f, 100, SAFE_DT, 99999.9f, AGENT_RADIUS, MAX_SPEED);

	sim->addAgent(RVO::Vector2(-5.0f, 0.1f));
	goals.push_back(RVO::Vector2(5.0f, 0.0f));
    sim->addAgent(RVO::Vector2(5.0f, -0.1f));
	goals.push_back(RVO::Vector2(-5.0f, 0.0f));
}

void updateVisualization(RVO::RVOSimulator *sim)
{
	/* Output the current global time. */
    std::cout << std::endl;

	/* Output the current position of all the agents. */
	for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {
        std::cout << sim->getGlobalTime();
		std::cout << " " << sim->getAgentPosition(i) << " " << sim->getAgentVelocity(i) << std::endl;
	}

	std::cout << std::endl;
}

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
	/*
	 * Set the preferred velocity to be a vector of unit magnitude (speed) in the
	 * direction of the goal.
	 */
	for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i)
    {
		RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);
        goalVector = RVO::normalize(goalVector) * MAX_SPEED;
		sim->setAgentPrefVelocity(i, goalVector);
	}
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
	/* Check if all agents have reached their goals. */
	for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {
		if (RVO::absSq(sim->getAgentPosition(i) - goals[i]) > sim->getAgentRadius(i) * sim->getAgentRadius(i))
        {
			return false;
		}
	}

	return true;
}

int main()
{
	/* Create a new simulator instance. */
	RVO::RVOSimulator *sim = new RVO::RVOSimulator();

	/* Set up the scenario. */
	setupScenarioTwoAgents(sim);

	/* Perform (and manipulate) the simulation. */
	do
    {
		updateVisualization(sim);
		setPreferredVelocities(sim);
		sim->doStep();
	}
	while (!reachedGoal(sim));

	delete sim;

	return 0;
}
