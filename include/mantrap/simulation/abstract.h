#ifndef MANTRAP_SIM_ABSTRACT_H
#define MANTRAP_SIM_ABSTRACT_H

#include <vector>

#include "mantrap/constants.h"
#include "mantrap/types.h"
#include "mantrap/agents/ados/dtvado.h"

namespace mantrap {

template<typename ego_t, typename prediction_t>
class Simulation {

public:

    mantrap::Axis _xaxis;
    mantrap::Axis _yaxis;
    double _dt;

    std::vector<mantrap::DTVAdo> _ados;
    ego_t _ego;

public:
    Simulation(const ego_t & ego,
               const mantrap::Axis & xaxis = mantrap::sim_x_axis_default,
               const mantrap::Axis & yaxis = mantrap::sim_y_axis_default,
               const double dt = mantrap::sim_dt_default)
    : _ego(ego), _xaxis(xaxis), _yaxis(yaxis), _dt(dt) {}

    // Predict the environments future for the given time horizon (discrete time).
    // The internal prediction model is dependent on the exact implementation of the internal interaction model
    // between the ados with each other and between the ados and the ego. The implementation therefore is specific
    // to each child-class.
    // Dependent on whether the prediction is deterministic or probablistic the output can vary between each child-
    // class, by setting the prediction_t. However the output should be a vector of predictions, one for each ado.
    // @param thorizon: number of timesteps to predict.
    // @param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
    virtual std::vector<prediction_t> predict(
            const int thorizon = mantrap::thorizon_default,
            const mantrap::Trajectory & ego_trajectory = mantrap::Trajectory()) const = 0;


    // Add another ado to the simulation.
    virtual void add_ado(const mantrap::DTVAdo & ado)
    {
        assert(ado.position().x > _xaxis.min && ado.position().x < _xaxis.max);
        assert(ado.position().y > _yaxis.min && ado.position().y < _yaxis.max);

        _ados.push_back(ado);
    }

    std::vector<mantrap::DTVAdo> ados() const           { return _ados; }
    ego_t ego() const                                   { return _ego; }
    double dt() const                                   { return _dt; }
    mantrap::Axis xaxis() const                         { return _xaxis; }
    mantrap::Axis yaxis() const                         { return _yaxis; }

};
}

#endif //MANTRAP_SIM_ABSTRACT_H
