#include "ros/ros.h"

#include "mantrap/agents/ados/single_mode.h"
#include "mantrap/simulation/environment.h"

#include "mantrap_ros/Scene.h"
#include "mantrap_ros/conversion.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "env_publisher");
    ros::NodeHandle n;
    ros::Publisher scene_pub = n.advertise<mantrap_ros::Scene>("scene", 1000);
    ros::Rate loop_rate(1);

    const mantrap::Position2D position(5, 2);
    const mantrap::Velocity2D v_mean(1, 0);
    Eigen::Matrix2d v_covariance;
    v_covariance << 1.0, 0.0, 0.0, 1.0;
    const mantrap::SingeModeDTVAdo ado(position, v_mean, v_covariance);

    const mantrap::Environment env;

    mantrap_ros::Scene scene_msg;
    scene_msg.env = mantrap_ros::env_2_msg(env);
    scene_msg.ados.push_back(mantrap_ros::single_mode_2_msg(ado));
    while (ros::ok()) {
        scene_pub.publish(scene_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
