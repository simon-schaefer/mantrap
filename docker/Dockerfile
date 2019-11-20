FROM ros:melodic
LABEL maintainer="Simon Schaefer sischaef@student.ethz.ch"

SHELL ["/bin/bash", "-c"]

# Create local catkin workspace.
ENV CATKIN_WS=/home/catkin_ws
RUN mkdir -p $CATKIN_WS/src

# Install wget to download external libraries.
RUN apt-get update && apt-get install -qq -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Install eigen library.
WORKDIR /home
ENV EIGEN_INCLUDE_DIR=/usr/local/include/eigen3
RUN wget -q http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
RUN tar -xf 3.3.7.tar.gz && rm 3.3.7.tar.gz
RUN mv eigen-eigen-323c052e1731 $EIGEN_INCLUDE_DIR

# Install catkin build and initialize local catkin workspace
WORKDIR $CATKIN_WS
RUN source /opt/ros/${ROS_DISTRO}/setup.bash
RUN apt-get update
RUN apt-get install python-catkin-tools -y
RUN rosdep install -y --from-paths . --ignore-src --rosdistro ${ROS_DISTRO}
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && catkin init && catkin build

# Starting commands (.bashrc).
COPY misc/banner.art /home/misc/banner.art
RUN echo "cat /home/misc/banner.art" >> ~/.bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source ${CATKIN_WS}/devel/setup.bash" >> ~/.bashrc

# Clear apt-cache to reduce image size.
RUN rm -rf /var/lib/apt/lists/*

# Working directory.
WORKDIR $CATKIN_WS

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]


# COPY . /home/catkin_ws
