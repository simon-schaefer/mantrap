#!/usr/bin/env python
from datetime import datetime
import os
import sys

import numpy as np

import mantrap_visualization

import rospy
import mantrap_ros.msg


OUTPUT_PATH = os.path.join(os.environ["PROJECT_PATH"], "outputs/plots", datetime.now().strftime("%d_%m_%Y @ %H_%M%_S"))


def _read_ego(data):
    ego_pose = np.asarray([data.ego.pose.x, data.ego.pose.y, data.ego.pose.theta])

    ego_history = []
    for pose in data.ego.history.poses:  # TODO: true theta from quaternion
        ego_history.append([pose.pose.position.x, pose.pose.position.y, 0, pose.header.stamp])
    ego_history = np.asarray(ego_history)

    ego_trajectory = []
    for pose in data.ego.trajectory.poses:  # TODO: true theta from quaternion
        ego_trajectory.append([pose.pose.position.x, pose.pose.position.y, 0, pose.header.stamp])
    ego_trajectory = np.asarray(ego_trajectory)

    return ego_pose, ego_history, ego_trajectory


def _read_ados(data):
    ados = []
    for ado_msg in data.ados:
        ado_pose = np.asarray([ado_msg.pose.x, ado_msg.pose.y, ado_msg.pose.theta])

        ado_history = []
        for pose in ado_msg.history.poses:  # TODO: true theta from quaternion
            ado_history.append([pose.pose.position.x, pose.pose.position.y, 0, pose.header.stamp])
        ado_history = np.asarray(ado_history)

        ado_trajectories = []
        for trajectory in ado_msg.trajectories:
            trajectory_data = []
            for pose in trajectory.poses:  # TODO: true theta from quaternion
                trajectory_data.append([pose.pose.position.x, pose.pose.position.y, 0, pose.header.stamp])
            ado_trajectories.append(trajectory_data)
        ado_trajectories = np.asarray(ado_trajectories)
        ados.append((ado_pose, ado_history, ado_trajectories))
    return ados


def callback(data):
    # Unparse message information.
    ego = _read_ego(data)
    ados = _read_ados(data)
    xaxis = (data.env.xmin.data, data.env.xmax.data)
    yaxis = (data.env.ymin.data, data.env.ymax.data)

    # Plotting.
    mantrap_visualization.plot_scene(ados, ego, xaxis, yaxis, OUTPUT_PATH)

    # Logging.
    rospy.loginfo("Stored plot in %s" % OUTPUT_PATH)
    rospy.signal_shutdown("")


def main():
    rospy.init_node("scene_visualizer", disable_signals=True)
    rospy.Subscriber("scene", mantrap_ros.msg.Scene, callback)
    rospy.spin()


if __name__ == '__main__':
    main()
