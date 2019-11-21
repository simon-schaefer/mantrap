#ifndef MANTRAP_ADOS_ABSTRACT_H
#define MANTRAP_ADOS_ABSTRACT_H

#include <vector>

#include <eigen3/Eigen/Core>

#include "gmm_stats/gmm_stats.h"
#include "mantrap/agents/agent.h"
#include "mantrap/constants.h"
#include "mantrap/types.h"

namespace mantrap {

template <class distribution_t>
class DTVAdo : mantrap::Agent {

protected:

    mantrap::Position2D _position;
    mantrap::Trajectory _history;
    int _num_modes;
    double _dt;

public:
    DTVAdo(const mantrap::Position2D & position,
           const mantrap::Trajectory& history,
           const int num_modes,
           const double dt = mantrap::sim_dt_default)
    : _num_modes(num_modes), _dt(dt)
    {
        assert(num_modes > 0);

        _history.resize(history.size());
        _history.insert(_history.begin(), history.begin(), history.end());
        _history.push_back(mantrap::PoseStamped2D(position));

        _position = position;
    }

    virtual distribution_t vpdf(const mantrap::Trajectory& history) const = 0;

    distribution_t vpdf_self() const             { return vpdf(_history);  }

    // Sample possible (future) trajectories given the currently stored, thorizon steps in the future.
    //
    // Therefore iteratively call the pdf() method, sample the next velocity from that pdf, forward integrate the
    // samples velocity to obtain the next position, append the history and repeat until the time horizon thorizon) is
    // reached, for each trajectory sample. Each trajectory starts with the last point of the objects history.
    // @param thorizon: length of trajectory samples, i.e. number of predicted time-steps.
    // @param num_samples: number of trajectory samples.
    // @return vector of sampled future trajectories (num_samples -> thorizon, 2).
    std::vector<mantrap::Trajectory> trajectory_samples(const int num_samples = 10,
                                                        const int thorizon = mantrap::thorizon_default) const
    {
        assert(num_samples > 0 && thorizon > 0);

        std::vector<mantrap::Trajectory> samples(num_samples);
        const mantrap::PoseStamped2D initial_pose(pose(), 0);
        for(int i = 0; i < num_samples; ++i) {
            mantrap::Trajectory sample_history = history();

            // First position in trajectory is current position.
            samples[i].resize(thorizon);
            samples[i][0] = initial_pose;

            // Obtain next position by forward integrating sample from internal velocity distribution, conditioned
            // on the full history (i.e. the internal history + previously sampled positions).
            mantrap::Velocity2D velocity_t;
            for(int t = 1; t < thorizon; ++t) {
                velocity_t = vpdf(sample_history).sample();
                samples[i][t].pose.x = samples[i][t-1].pose.x + velocity_t.vx * _dt;
                samples[i][t].pose.y = samples[i][t-1].pose.y + velocity_t.vy * _dt;
                samples[i][t].t = samples[i][t-1].t + _dt;
                sample_history.push_back(samples[i][t]);
            }
        }
        return samples;
    }

    int num_modes() const                   { return _num_modes; }
    mantrap::Position2D position() const    { return _position; }
    mantrap::Pose2D pose() const            { return mantrap::Pose2D(_position.x, _position.y, 0); }
    mantrap::Trajectory history() const     { return _history; }

};
}

#endif //MANTRAP_ADOS_ABSTRACT_H
