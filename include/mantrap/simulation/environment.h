#ifndef MANTRAP_ENVIRONMENT_H
#define MANTRAP_ENVIRONMENT_H

#include <any>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "mantrap/agents/ados/abstract.h"
#include "mantrap/agents/egos/abstract.h"


namespace mantrap {

class Environment {

    Eigen::Vector2d _xaxis;
    Eigen::Vector2d _yaxis;
    double _dt;

    std::vector<std::any> _ados;
    std::any _ego;

public:
    Environment(const Eigen::Vector2d xaxis = Eigen::Vector2d{0, 10},
                const Eigen::Vector2d yaxis = Eigen::Vector2d{0, 10},
                const double dt = 0.1);

    // Generate trajectory samples for each ado in the environment.
    // Therefore iterate over all internally stored ados and call the generate sampling function. The number of
    // samples thereby describes the number of generated samples per ados, i.e. the total size of the (nested)
    // returned vector is num_ados * num_samples, while each trajectory has thorizon length.
    // @param thorizon: length of trajectory samples, i.e. number of predicted time-steps.
    // @param num_samples: number of trajectory samples per ado.
    // @return vector of sampled future trajectories (num_samples -> thorizon, 2).
    std::vector<std::vector<mantrap::Trajectory>> generate_trajectory_samples(const int thorizon = 20,
                                                                              const int num_samples = 10) const;

    void add_ego(const std::any & ego);
    void add_ado(const std::any & ado);

    std::vector<std::any> ados() const                  { return _ados; }
    std::any ego() const                                { return _ego; }
    double dt() const                                   { return _dt; }
};
}


#endif //MANTRAP_ENVIRONMENT_H
