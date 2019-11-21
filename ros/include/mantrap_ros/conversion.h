#ifndef MANTRAP_CONVERSION_H
#define MANTRAP_CONVERSION_H

#include <cmath>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "ros/ros.h"
#include "geometry_msgs/Quaternion.h"

#include "mantrap/types.h"
#include "mantrap/agents/ados/single_mode.h"
#include "mantrap/simulation/environment.h"

#include "mantrap_ros/Ado.h"
#include "mantrap_ros/Environment.h"


namespace mantrap_ros {

// Transform 2D orientation angle (yaw -> theta) to quaternion message, following:
// http://www.weizmann.ac.il/sci-tea/benari/sites/sci-tea.benari/files/uploads/softwareAndLearningMaterials/quaternion-tutorial-2-0-1.pdf
geometry_msgs::Quaternion theta_2_quaternion(const double theta) {
    geometry_msgs::Quaternion quat;
    quat.x = cos(theta / 2);
    quat.y = 0;
    quat.z = 0;
    quat.w = sin(theta / 2);
    return quat;
}


mantrap_ros::Environment env_2_msg(const mantrap::Environment& env)
{
    mantrap_ros::Environment env_msg;

    const mantrap::Axis xaxis = env.xaxis();
    const mantrap::Axis yaxis = env.yaxis();
    env_msg.xmin.data = xaxis.min;
    env_msg.xmax.data = xaxis.max;
    env_msg.ymin.data = yaxis.min;
    env_msg.ymax.data = yaxis.max;

    return env_msg;
}


mantrap_ros::Ado single_mode_2_msg(const mantrap::SingeModeDTVAdo& ado)
{
    mantrap_ros::Ado ado_msg;
    const ros::Time time_ros = ros::Time::now();

    ado_msg.header.stamp = time_ros;

    ado_msg.pose.x = ado.pose().x;
    ado_msg.pose.y = ado.pose().y;
    ado_msg.pose.theta = ado.pose().theta;

    ado_msg.history.header.stamp = time_ros;
    ado_msg.history.poses.resize(ado.history().size());
    for(int k = 0; k < ado.history().size(); ++k) {
        ado_msg.history.poses[k].header.stamp = time_ros + ros::Duration(ado.history()[k].t);
        ado_msg.history.poses[k].pose.position.x = ado.history()[k].pose.x;
        ado_msg.history.poses[k].pose.position.y = ado.history()[k].pose.y;
        ado_msg.history.poses[k].pose.position.z = 0;
        ado_msg.history.poses[k].pose.orientation = theta_2_quaternion(ado.history()[k].pose.theta);
    }

    const std::vector<mantrap::Trajectory> samples = ado.trajectory_samples();
    ado_msg.trajectories.resize(samples.size());
    for(int i = 0; i < samples.size(); ++i) {
        ado_msg.trajectories[i].header.stamp = time_ros;
        ado_msg.trajectories[i].poses.resize(samples[i].size());
        for(int k = 0; k < samples[i].size(); ++k) {
            ado_msg.trajectories[i].poses[k].header.stamp = time_ros + ros::Duration(samples[i][k].t);
            ado_msg.trajectories[i].poses[k].pose.position.x = samples[i][k].pose.x;
            ado_msg.trajectories[i].poses[k].pose.position.y = samples[i][k].pose.y;
            ado_msg.trajectories[i].poses[k].pose.position.z = 0;
            ado_msg.trajectories[i].poses[k].pose.orientation = theta_2_quaternion(samples[i][k].pose.theta);
        }
    }
    return ado_msg;
}

}

#endif //MANTRAP_CONVERSION_H
