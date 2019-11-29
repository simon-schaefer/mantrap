#include <vector>

#include "ros/ros.h"

#include "mantrap/agents/ados/dtvado.h"
#include "mantrap/agents/egos/integrator.h"
#include "mantrap/simulation/social_forces.h"

#include "mantrap_ros/Scene.h"
#include "mantrap_ros/conversion.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scene_publisher");
    ros::NodeHandle n;
    ros::Publisher scene_pub = n.advertise<mantrap_ros::Scene>("scene", 1000);
    ros::Rate loop_rate(1);

    const mantrap::IntegratorDTEgo ego(-5, 5);
    mantrap::SocialForcesSimulation sim(ego);
    const mantrap::DTVAdo ado_1(-5, 0.1, 0.2, 0.2);
    const mantrap::Vector2D ado_1_goal(5, 0);
    sim.add_ado(ado_1, ado_1_goal);
    const mantrap::DTVAdo ado_2(5, -0.1, -0.5, 0.2);
    const mantrap::Vector2D ado_2_goal(-5, 0);
    sim.add_ado(ado_2, ado_2_goal);

    mantrap_ros::Scene scene_msg = mantrap_ros::social_sim_2_msg<mantrap::IntegratorDTEgo>(sim);
    while (ros::ok())
    {
        scene_pub.publish(scene_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
