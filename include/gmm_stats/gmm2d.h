#ifndef GMM_STATS_GMM2D_H
#define GMM_STATS_GMM2D_H

#include <vector>

#include <eigen3/Eigen/Dense>

#include "gmm_stats/gaussian2d.h"
#include "gmm_stats/types.h"

/* Gaussian Mixture model distribution:
f(x) = sum_i w_i * Gaussian2D_i(x) */
namespace gmmstats {

class GMM2D : public gmmstats::Distribution2D {

    std::vector<Gaussian2D> _gaussians;
    std::vector<double> _weights;

public:
    GMM2D(const gmmstats::Matrix2Xd & means,
          const gmmstats::Matrix22Xd & covariances,
          const std::vector<double> & weights);

    // Get probability density function value at given location.
    // @param position: 2D position vector to sample at.
    // @return pdf value for given (x,y)-pair.
    double pdfAt(const Eigen::Vector2d & position) const;

    // Get probability density function value at given locations.
    // The x, y coordinates for a vector of (x, y) pairs, the function is shape containing, i.e. it returns
    // probability values with the same shape as x and y (MxN). Since the evaluation of the pdf function is based
    // on (x, y) pairs hence x and y have to be in the same shape.
    // @param positions: std::vector of 2D position vectors to sample at.
    // @return pdf values for given (x,y)-pair.
    std::vector<double> pdfAt(const std::vector<Eigen::Vector2d> & positions) const;

    // Sample a random 2D position from GMM.
    // For this model the modes as (weighted) discrete distribution, weighted by the GMM weights, and sample a
    // mode index from this distribution. Then sample from choosen mode.
    Eigen::Vector2d sample() const;

    // Sample N = num_samples from distribution and return results stacked in x and y std::vector.
    // @param num_samples: number of samples to return.
    // @return std::vector of 2D position vectors.
    std::vector<Eigen::Vector2d> sample(const int num_samples) const;

    gmmstats::GMM2D& operator+(const GMM2D& other);

    gmmstats::Gaussian2D mode(const int mode_id) const;
    gmmstats::Matrix2Xd mus() const;
    gmmstats::Matrix22Xd covariances() const;

    std::vector<double> weights() const         { return _weights;  }
    std::vector<Gaussian2D> gaussians() const   { return _gaussians; }

};
}



#endif //GMM_STATS_GMM2D_H
