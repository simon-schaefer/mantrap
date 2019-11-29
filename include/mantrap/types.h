#ifndef MANTRAP_TYPES_H
#define MANTRAP_TYPES_H

#include <cmath>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace mantrap {

    struct Vector2D
    {
        double x;
        double y;

        Vector2D() : x(0), y(0) {}
        Vector2D(const double x_, const double y_) : x(x_), y(y_) {}

        Eigen::Vector2d to_eigen() const    { return Eigen::Vector2d{x, y}; }
        void from_eigen(const Eigen::Vector2d& vec)
        {
            assert(vec.size() == 2);
            x = vec(0);
            y = vec(1);
        }

        double norml2() const               { return sqrt(pow(x, 2) + pow(y, 2)); }

        mantrap::Vector2D normalize() const
        {
            const double norm = norml2();
            if(norm <= 1e-5)
            {
                return mantrap::Vector2D(x, y);
            }
            else
            {
                return mantrap::Vector2D(x / norm, y / norm);
            }
        }

        Vector2D operator+(const Vector2D& other) const
        {
            const double x_ = x + other.x;
            const double y_ = y + other.y;
            return mantrap::Vector2D(x_, y_);
        }

        Vector2D operator-(const Vector2D& other) const
        {
            const double x_ = x - other.x;
            const double y_ = y - other.y;
            return mantrap::Vector2D(x_, y_);
        }

        Vector2D operator*(const double& scalar) const
        {
            const double x_ = x * scalar;
            const double y_ = y * scalar;
            return mantrap::Vector2D(x_, y_);
        }

        Vector2D operator/(const double& scalar) const
        {
            const double x_ = x / scalar;
            const double y_ = y / scalar;
            return mantrap::Vector2D(x_, y_);
        }

        void print() const
        {
            std::cout << "x: " << x << std::endl;
            std::cout << "y: " << y << std::endl;
        }

    };

    // navigation containers.

    struct Pose2D
    {
       double x;
       double y;
       double theta;

       Pose2D() : x(0), y(0), theta(0) {}
       Pose2D(const double x_, const double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
       Pose2D(const double x_, const double y_) : x(x_), y(y_), theta(0) {}
       Pose2D(const Eigen::Vector2d & vec) : x(vec(0)), y(vec(1)), theta(0) {}
       Pose2D(const Eigen::Vector3d & vec) : x(vec(0)), y(vec(1)), theta(vec(2)) {}

       Eigen::Vector3d to_eigen() const { return Eigen::Vector3d{x, y, theta}; }

       void print() const
       {
           std::cout << "x: " << x << std::endl;
           std::cout << "y: " << y << std::endl;
           std::cout << "theta: " << theta << std::endl;
       }
    };

    struct PoseStamped2D
    {
        Pose2D pose;
        double t;

        PoseStamped2D() : pose(), t(0) {}
        PoseStamped2D(const mantrap::Pose2D & pose_, const double t_) : pose(pose_), t(t_) {}
        PoseStamped2D(const Eigen::Vector2d& position) : pose(position), t(0) {}
        PoseStamped2D(const mantrap::Vector2D position) : pose(position.x, position.y), t(0) {}
        PoseStamped2D(const double x, const double y, const double theta, const double t) : pose(x, y, theta), t(t) {}

        void print() const
        {
            std::cout << "pose: " << std::endl;
            pose.print();
            std::cout << "t: " << t << std::endl;
        }
    };

    typedef std::vector<PoseStamped2D> Trajectory;

    // environment utility containers.

    struct Axis
    {
        double min;
        double max;

        Axis(double _min, double _max) : min(_min), max(_max) {}
    };
}

#endif //MANTRAP_TYPES_H
