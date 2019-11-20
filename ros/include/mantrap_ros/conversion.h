#ifndef MANTRAP_CONVERSION_H
#define MANTRAP_CONVERSION_H

#include <iostream>

#include "mantrap/agents/ados/single_mode.h"

#include "mantrap_ros/Ado.h"

namespace mantrap_ros {

mantrap_ros::Ado single_mode_2_msg(const mantrap::SingeModeDTVAdo& ado)
{
    mantrap_ros::Ado ado_msg;

    ado_msg.pose.x = ado.position()(0);
    ado_msg.pose.y = ado.position()(1);

    std::cout << ado.history().size() << std::endl;

    ado_msg.history.poses.resize(ado.history().size());
    for(int k = 0; k < ado.history().size(); ++k) {
        std::cout << ado.history()[k] << std::endl;
        ado_msg.history.poses[k].pose.position.x = ado.history()[k](0);
        ado_msg.history.poses[k].pose.position.y = ado.history()[k](1);
    }

    const std::vector<mantrap::Trajectory> samples = ado.trajectory_samples(2, 1);
    ado_msg.trajectories.resize(samples.size());
    for(int i = 0; i < samples.size(); ++i) {
        ado_msg.trajectories[i].poses.resize(samples[i].size());
        for(int k = 0; k < samples[i].size(); ++k) {
            ado_msg.trajectories[i].poses[k].pose.position.x = samples[i][k](0);
            ado_msg.trajectories[i].poses[k].pose.position.y = samples[i][k](1);
        }
    }
    return ado_msg;
}
}

#endif //MANTRAP_CONVERSION_H
