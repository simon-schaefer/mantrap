#ifndef MANTRAP_TYPES_H
#define MANTRAP_TYPES_H

#include <vector>

#include <eigen3/Eigen/Dense>

namespace mantrap {

    struct Position2D {
        double x;
        double y;

        Position2D() : x(0), y(0) {}
        Position2D(const double x_, const double y_) : x(x_), y(y_) {}
        Position2D(const Position2D& other) : x(other.x), y(other.y) {}
        Position2D(const Eigen::Vector2d & vec) : x(vec(0)), y(vec(1)) {}

        Eigen::Vector2d to_eigen() const { return Eigen::Vector2d{x, y}; }
    };

    struct Pose2D {
       double x;
       double y;
       double theta;

       Pose2D() : x(0), y(0), theta(0) {}
       Pose2D(const double x_, const double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
       Pose2D(const double x_, const double y_) : x(x_), y(y_), theta(0) {}
       Pose2D(const Eigen::Vector2d & vec) : x(vec(0)), y(vec(1)), theta(0) {}
       Pose2D(const Eigen::Vector3d & vec) : x(vec(0)), y(vec(1)), theta(vec(2)) {}

        Eigen::Vector3d to_eigen() const { return Eigen::Vector3d{x, y, theta}; }
    };

    struct PoseStamped2D {
        Pose2D pose;
        double t;

        PoseStamped2D() : pose(), t(0) {}
        PoseStamped2D(const mantrap::Pose2D & pose_, const double t_) : pose(pose_), t(t_) {}
        PoseStamped2D(const Eigen::Vector2d& position) : pose(position), t(0) {}
        PoseStamped2D(const mantrap::Position2D position) : pose(position.x, position.y), t(0) {}
    };

    struct Velocity2D {
        double vx;
        double vy;

        Velocity2D() : vx(0), vy(0) {}
        Velocity2D(const double vx_, const double vy_) : vx(vx_), vy(vy_) {}
        Velocity2D(const Eigen::Vector2d & vec) : vx(vec(0)), vy(vec(1)) {}

        Eigen::Vector2d to_eigen() const { return Eigen::Vector2d{vx, vy}; }
    };

    typedef std::vector<PoseStamped2D> Trajectory;
}

#endif //MANTRAP_TYPES_H
