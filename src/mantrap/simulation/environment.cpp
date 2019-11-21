#include "mantrap/simulation/environment.h"
#include "mantrap/agents/ados/single_mode.h"

mantrap::Environment::Environment(const mantrap::Axis & xaxis, const mantrap::Axis & yaxis, const double dt)
: _xaxis(xaxis), _yaxis(yaxis), _dt(dt) {}


std::vector<std::vector<mantrap::Trajectory>>
mantrap::Environment::generate_trajectory_samples(const int thorizon,
                                                  const int num_samples) const
{
    const int num_ados = ados().size();
    std::vector<std::vector<mantrap::Trajectory>> samples(num_ados);
    for(int i = 0; i < num_ados; ++i) {
        samples[i].resize(num_samples);
        samples[i] = std::any_cast<mantrap::SingeModeDTVAdo>(_ados[i]).trajectory_samples(thorizon, num_samples);
    }
    return samples;
}


void
mantrap::Environment::add_ego(const std::any & ego)
{
    _ego = ego;
}


void
mantrap::Environment::add_ado(const std::any & ado)
{
    _ados.push_back(ado);
}