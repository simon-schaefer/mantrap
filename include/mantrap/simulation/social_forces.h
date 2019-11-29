#ifndef MANTRAP_SIM_SOCIAL_FORCES_H
#define MANTRAP_SIM_SOCIAL_FORCES_H

#include <iostream>
#include <vector>

#include "mantrap/types.h"
#include "mantrap/simulation/abstract.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Simulation based on Social Forces Interaction model ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mantrap {

template<typename ego_t>
class SocialForcesSimulation : public Simulation<ego_t, mantrap::Trajectory>
{

protected:

    std::vector<mantrap::Vector2D> _ado_goals;
    typedef mantrap::Vector2D Force2D;

public:
    SocialForcesSimulation(const ego_t & ego,
                           const mantrap::Axis & xaxis = mantrap::sim_x_axis_default,
                           const mantrap::Axis & yaxis = mantrap::sim_y_axis_default,
                           const double dt = mantrap::sim_dt_default)
    : mantrap::Simulation<ego_t, mantrap::Trajectory>(ego, xaxis, yaxis, dt) {}

    // Compute the trajectory for each ado in the environment.
    // Therefore apply the social forces model described in "Social force model for pedestrian dynamics" (Helbing),
    // only taking the destination forces and interactive forces into account.
    std::vector<mantrap::Trajectory> predict(
            const int thorizon = mantrap::thorizon_default,
            const mantrap::Trajectory & ego_trajectory = mantrap::Trajectory()) const
    {
        // Define simulation parameters (as defined in the paper).
        const int num_ados = this->_ados.size();
        const double tau = 0.5;  // [s] relaxation time (assumed to be uniform over all agents).
        const double V_0 = 2.1;  // [m2s-2] repulsive field constant.
        const double sigma = 0.6;  // [m] repuslive field exponent constant.
        const double m = 1;  // [kg] mass of ado for velocity update.

        // The social forces model predicts from one timestep to another, therefore the ados are actually updated in each
        // time step, in order to predict the next timestep. To not change the initial state, hence, the ados vector
        // is copied.
        std::vector<mantrap::DTVAdo> ados_sim = this->ados();
        const double dt = this->dt();

        for(int t = 0; t < thorizon; ++t)
        {
            // Compute summed 2D force vector for each agent.
            for(int j = 0; j < num_ados; ++j)
            {
                Force2D force(0, 0);

                // Destination force - Force pulling the ado to its assigned goal position.
                const mantrap::Vector2D alpha_direction = (_ado_goals[j] - ados_sim[j].position()).normalize();
                force = (alpha_direction * ados_sim[j].speed() - ados_sim[j].velocity()) * 1 / tau;

                // Interactive force - Repulsive potential field by every other agent.
                // Gradient formula according to https://github.com/bullbo/group-based-crowd-simulation/Agent.py.
                for(int l = 0; l < num_ados; ++l)
                {
                    if(j == l)
                    {
                        continue;
                    }

                    const mantrap::Vector2D beta_direction = (_ado_goals[l] - ados_sim[l].position()).normalize();
                    const mantrap::Vector2D relative_distance = ados_sim[j].position() - ados_sim[l].position();
                    const double norm_relative_distance = relative_distance.norml2();
                    const mantrap::Vector2D relative_velocity = ados_sim[j].velocity() - ados_sim[l].velocity();
                    const double  norm_relative_velocity = relative_velocity.norml2();
                    const mantrap::Vector2D diff_position = relative_distance - relative_velocity * dt;
                    const double norm_diff_position = diff_position.norml2();

                    const double b1 = norm_relative_distance + diff_position.norml2();
                    const double b2 = dt * norm_relative_velocity;
                    const double b = 0.5 * sqrt(pow(b1, 2) - pow(b2, 2));

                    const double f1 = V_0 * exp(-b / sigma);
                    const double f2 = (norm_relative_distance + norm_diff_position) / 4 * b;
                    const mantrap::Vector2D f3 =  relative_distance.normalize() + diff_position.normalize();
                    force = force - f3 * f1 * f2;
                }

                // Update ados velocity (single integrator dynamics) and position.
                ados_sim[j].update(force / m, dt * t);
            }

        }

        // Collect histories of simulated ados (last thorizon steps are equal to future trajectories).
        std::vector<mantrap::Trajectory> trajectories(num_ados);
        for(int j = 0; j < num_ados; ++j)
        {
            trajectories[j] = ados_sim[j].history();
        }
        return trajectories;
    }

    // Add another ado to the simulation. In the social forces model every agent is assigned to some goal,
    // so next to the ado itself this goal position has to be added.
    void add_ado(const mantrap::DTVAdo & ado, const mantrap::Vector2D & goal_position)
    {
        _ado_goals.push_back(goal_position);
        Simulation<ego_t, mantrap::Trajectory>::add_ado(ado);
    }

};
}

#endif //MANTRAP_SIM_SOCIAL_FORCES_H
