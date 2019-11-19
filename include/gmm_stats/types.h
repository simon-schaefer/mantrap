#ifndef GMM_STATS_TYPES_H
#define GMM_STATS_TYPES_H

#include <vector>

#include <eigen3/Eigen/Dense>

namespace gmmstats {
    typedef std::vector<Eigen::Vector2d> Matrix2Xd;
    typedef std::vector<Eigen::Matrix2d> Matrix22Xd;

    class Distribution2D {

    public:
        virtual double pdfAt(const Eigen::Vector2d & position) const = 0;
        virtual std::vector<double> pdfAt(const std::vector<Eigen::Vector2d> & positions) const = 0;

        virtual Eigen::Vector2d sample() const = 0;
        virtual std::vector<Eigen::Vector2d> sample(int num_samples) const = 0;
    };

}

#endif //GMM_STATS_TYPES_H
