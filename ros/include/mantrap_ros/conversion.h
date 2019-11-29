#ifndef MANTRAP_CONVERSION_H
#define MANTRAP_CONVERSION_H

#include <cmath>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "ros/ros.h"
#include "geometry_msgs/Quaternion.h"

#include "mantrap/types.h"
#include "mantrap/simulation/social_forces.h"

#include "mantrap_ros/Ado.h"
#include "mantrap_ros/Environment.h"


namespace mantrap_ros {

// Transform 2D orientation angle (yaw -> theta) to quaternion message, following:
// http://www.weizmann.ac.il/sci-tea/benari/sites/sci-tea.benari/files/uploads/softwareAndLearningMaterials/quaternion-tutorial-2-0-1.pdf
geometry_msgs::Quaternion theta_2_quaternion(const double theta)
{
    geometry_msgs::Quaternion quat;
    quat.x = cos(theta / 2);
    quat.y = 0;
    quat.z = 0;
    quat.w = sin(theta / 2);
    return quat;
}


template<typename ego_t>
mantrap_ros::Scene social_sim_2_msg(const mantrap::SocialForcesSimulation<ego_t>& sim,
                                    const mantrap::Trajectory& ego_trajectory = mantrap::Trajectory())
{
    mantrap_ros::Scene scene_msg;

    // Environment message.
    const mantrap::Axis xaxis = sim.xaxis();
    const mantrap::Axis yaxis = sim.yaxis();
    scene_msg.env.xmin.data = xaxis.min;
    scene_msg.env.xmax.data = xaxis.max;
    scene_msg.env.ymin.data = yaxis.min;
    scene_msg.env.ymax.data = yaxis.max;

    // Ego message.
    mantrap_ros::Ego ego_msg;
    const ros::Time time_ros = ros::Time::now();
    ego_msg.header.stamp = time_ros;

    ego_msg.pose.x = sim.ego().pose().x;
    ego_msg.pose.y = sim.ego().pose().y;
    ego_msg.pose.theta = sim.ego().pose().theta;

    ego_msg.history.header.stamp = time_ros;
    ego_msg.history.poses.resize(sim.ego().history().size());
    for(int k = 0; k < sim.ego().history().size(); ++k)
    {
        ego_msg.history.poses[k].header.stamp = time_ros + ros::Duration(sim.ego().history()[k].t);
        ego_msg.history.poses[k].pose.position.x = sim.ego().history()[k].pose.x;
        ego_msg.history.poses[k].pose.position.y = sim.ego().history()[k].pose.y;
        ego_msg.history.poses[k].pose.position.z = 0;
        ego_msg.history.poses[k].pose.orientation = theta_2_quaternion(sim.ego().history()[k].pose.theta);
    }

    ego_msg.trajectory.header.stamp = time_ros;
    ego_msg.trajectory.poses.resize(ego_trajectory.size());
    for(int k = 0; k < ego_trajectory.size(); ++k)
    {
        ego_msg.trajectory.poses[k].header.stamp = time_ros + ros::Duration(ego_trajectory[k].t);
        ego_msg.trajectory.poses[k].pose.position.x = ego_trajectory[k].pose.x;
        ego_msg.trajectory.poses[k].pose.position.y = ego_trajectory[k].pose.y;
        ego_msg.trajectory.poses[k].pose.position.z = 0;
        ego_msg.trajectory.poses[k].pose.orientation = theta_2_quaternion(ego_trajectory[k].pose.theta);
    }
    scene_msg.ego = ego_msg;

    // Ado message.
    for(int k = 0; k < sim.ados().size(); ++k)
    {
        const mantrap::DTVAdo ado = sim.ados()[k];

        mantrap_ros::Ado ado_msg;
        const ros::Time time_ros = ros::Time::now();

        ado_msg.header.stamp = time_ros;

        ado_msg.pose.x = ado.pose().x;
        ado_msg.pose.y = ado.pose().y;
        ado_msg.pose.theta = ado.pose().theta;

        ado_msg.history.header.stamp = time_ros;
        ado_msg.history.poses.resize(ado.history().size());
        for(int k = 0; k < ado.history().size(); ++k)
        {
            ado_msg.history.poses[k].header.stamp = time_ros + ros::Duration(ado.history()[k].t);
            ado_msg.history.poses[k].pose.position.x = ado.history()[k].pose.x;
            ado_msg.history.poses[k].pose.position.y = ado.history()[k].pose.y;
            ado_msg.history.poses[k].pose.position.z = 0;
            ado_msg.history.poses[k].pose.orientation = theta_2_quaternion(ado.history()[k].pose.theta);
        }

        const std::vector<mantrap::Trajectory> samples = sim.predict();
        ado_msg.trajectories.resize(samples.size());
        for(int i = 0; i < samples.size(); ++i)
        {
            ado_msg.trajectories[i].header.stamp = time_ros;
            ado_msg.trajectories[i].poses.resize(samples[i].size());
            for(int k = 0; k < samples[i].size(); ++k)
            {
                ado_msg.trajectories[i].poses[k].header.stamp = time_ros + ros::Duration(samples[i][k].t);
                ado_msg.trajectories[i].poses[k].pose.position.x = samples[i][k].pose.x;
                ado_msg.trajectories[i].poses[k].pose.position.y = samples[i][k].pose.y;
                ado_msg.trajectories[i].poses[k].pose.position.z = 0;
                ado_msg.trajectories[i].poses[k].pose.orientation = theta_2_quaternion(samples[i][k].pose.theta);
            }
        }
        scene_msg.ados.push_back(ado_msg);
    }

    return scene_msg;
}

}

#endif //MANTRAP_CONVERSION_H
