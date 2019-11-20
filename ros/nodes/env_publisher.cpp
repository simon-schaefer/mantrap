#include "ros/ros.h"

#include "mantrap/agents/ados/single_mode.h"

#include "mantrap_ros/Ado.h"
#include "mantrap_ros/conversion.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "env_publisher");
    ros::NodeHandle n;
    ros::Publisher ado_pub = n.advertise<mantrap_ros::Ado>("ado", 1000);
    ros::Rate loop_rate(10);

    const mantrap::Position2D position(-5, 1);
    const mantrap::Velocity2D v_mean(1, 0);
    Eigen::Matrix2d v_covariance;
    v_covariance << 1.0, 0.0, 0.0, 1.0;
    const mantrap::SingeModeDTVAdo ado(position, v_mean, v_covariance);

    mantrap_ros::Ado ado_msg = mantrap_ros::single_mode_2_msg(ado);

    while (ros::ok()) {
        ado_pub.publish(ado_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
