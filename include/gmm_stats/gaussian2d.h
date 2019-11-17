#ifndef GMM_STATS_Gaussian2D_H
#define GMM_STATS_Gaussian2D_H

#include <vector>

#include <eigen3/Eigen/Dense>

/* Unimodal Gaussian distribution:
f(x) = 1 / /sqrt(2*pi )^p * det(Sigma)) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu)) */
class Gaussian2D {

    double _K1, _K2;

public:

    Eigen::Vector2d mu;
    Eigen::Matrix2d sigma;

    Gaussian2D(const Eigen::Vector2d & mean, const Eigen::Matrix2d & covariance);

    // Get probability density function value at given location.
    // @param position: 2D position vector to sample at.
    // @return pdf value for given (x,y)-pair.
    [[nodiscard]] double pdf_at(const Eigen::Vector2d & position) const;

    // Get probability density function value at given locations.
    // The x, y coordinates for a vector of (x, y) pairs, the function is shape containing, i.e. it returns
    // probability values with the same shape as x and y (MxN). Since the evaluation of the pdf function is based
    // on (x, y) pairs hence x and y have to be in the same shape.
    // @param positions: std::vector of 2D position vectors to sample at.
    // @return pdf values for given (x,y)-pair.
    std::vector<double> pdf_at(const std::vector<Eigen::Vector2d> & positions) const;

    // Sample a value from 2D distribution.
    // For this, we basically produce data from the normal distribution around zero with identity convariance matrix
    // and then we transform this data to the given statistics. Based on the Central Limit Theorem we can produce a
    // random number x, that belongs to the Guassian Distribution N(0, 1) by producing p random numbers k which belong
    // to the uniform distribution U(0, 1):clt
    //
    // x = (sum_i=1^p k_i - p/2) / sqrt(p/12)
    //
    // The bigger the number p the better we approximate the guassian distribution. Thus, if we produce a vector of n
    // such numbers this vector will be a point belonging to the Multivariate Guassian Distribution N(0, I), i.e.
    // with mean the zero vector and convariance the identity matrix. These random vectors will be a high-dimensional
    // sphere around zero. We can transform this vectors x to vectors y which belong to any MVN with any mean and
    // covariance matrix by:
    //
    // y = lambda**(1/2) * phi * x + mu
    //
    // where lambda a diagonal matrix with the eigenvalues of the convariance matrix as the diagonal elements and phi
    // matrix with the eigenvectors of the convariance matrix at its column.
    // Inspired by http://blog.sarantop.com/notes/mvn.
    Eigen::Vector2d sample(int num_iterations = 200) const;

    // Sample N = num_samples from distribution and return results stacked in x and y std::vector.
    // @param num_samples: number of samples to return.
    // @return std::vector of 2D position vectors.
    std::vector<Eigen::Vector2d> sample_vector(const int num_samples) const;

};


#endif //GMM_STATS_Gaussian2D_H
