#ifndef MANTRAP_ADOS_ABSTRACT_H
#define MANTRAP_ADOS_ABSTRACT_H

#include <vector>

#include "gmm_stats/gmm_stats.h"
#include "mantrap/agents/agent.h"
#include "mantrap/types.h"

namespace mantrap {

template <class distribution_t>
class DTVAdo : mantrap::Agent {

protected:

    mantrap::Trajectory _history;
    int _num_modes;

public:
    DTVAdo(const mantrap::Trajectory& history, const int num_modes)
    : _num_modes(num_modes) {
        assert(num_modes > 0);

        _history.resize(history.size());
        _history.insert(_history.begin(), history.begin(), history.end());
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
    // @param dt: time step [s] for forward integration.
    // @return vector of sampled future trajectories (num_samples -> thorizon, 2).
    std::vector<mantrap::Trajectory> trajectory_samples(const int thorizon,
                                                        const int num_samples,
                                                        const double dt = 0.1) const  {
        assert(dt > 0 && num_samples > 0 && thorizon > 0);

        std::vector<mantrap::Trajectory> samples(num_samples);
        for(int i = 0; i < num_samples; ++i) {
            mantrap::Trajectory sample_history = history();

            // First position in trajectory is current position.
            samples[i].resize(thorizon);
            samples[i][0] = position();

            // Obtain next position by forward integrating sample from internal velocity distribution, conditioned
            // on the full history (i.e. the internal history + previously sampled positions).
            mantrap::Velocity2D velocity_t;
            mantrap::Position2D position_t;
            for(int t = 1; t < thorizon; ++t) {
                velocity_t = vpdf(sample_history).sample();
                position_t = samples[i][t-1] + velocity_t * dt;
                samples[i][t] = position_t;
                sample_history.push_back(position_t);
            }

        }
        return samples;
    }

    int num_modes() const                   { return _num_modes; }
    mantrap::Position2D position() const    { return *_history.begin(); }
    mantrap::Trajectory history() const     { return _history; }

};
}

#endif //MANTRAP_ADOS_ABSTRACT_H
