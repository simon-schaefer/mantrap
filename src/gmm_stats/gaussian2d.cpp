#include <cassert>
#include <cmath>

#include "gmm_stats/gaussian2d.h"

#define SAMPLING_NUM_ITERATIONS 200

gmmstats::Gaussian2D::Gaussian2D() {
    _mu << 0, 0;
    _sigma << 1, 0, 0, 1;
}


gmmstats::Gaussian2D::Gaussian2D(const Eigen::Vector2d & mean, const Eigen::Matrix2d & covariance)
: _mu(mean), _sigma(covariance) {
    assert(covariance.determinant() != 0);
    assert(covariance(0, 1) == covariance(1, 0));
}


double gmmstats::Gaussian2D::pdfAt(const Eigen::Vector2d & position) const {
    const double sqrt2pi = std::sqrt(2 * M_PI);
    const double quadform  = (position - _mu).transpose() * _sigma.inverse() * (position - _mu);
    const double norm = std::pow(sqrt2pi, - 2) * std::pow(_sigma.determinant(), - 0.5);
    return norm * exp(-0.5 * quadform);
}


std::vector<double> gmmstats::Gaussian2D::pdfAt(const std::vector<Eigen::Vector2d> & positions) const {
    std::vector<double> pdfs(positions.size());
    for(int i = 0; i < positions.size(); i++) {
        pdfs[i] = pdfAt(positions[i]);
    }
    return pdfs;
}


Eigen::Vector2d gmmstats::Gaussian2D::sample() const {
    // Generate x from the N(0, I) distribution
    Eigen::Vector2d x;
    Eigen::Vector2d sum;
    sum.setZero();
    for(unsigned int i = 0; i < SAMPLING_NUM_ITERATIONS; i++) {
        x.setRandom();
        x = 0.5 * (x + Eigen::VectorXd::Ones(2));
        sum = sum + x;
    }
    sum = sum - (static_cast<double>(SAMPLING_NUM_ITERATIONS) / 2) * Eigen::VectorXd::Ones(2);
    x = sum / (std::sqrt(static_cast<double>(SAMPLING_NUM_ITERATIONS) / 12));

    // Find the eigen vectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(_sigma);
    Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

    // Find the eigenvalues of the covariance matrix
    Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

    // Find the transformation matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
    Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
    Eigen::MatrixXd Q = eigenvectors * sqrt_eigenvalues;

    return Q * x + _mu;
}


std::vector<Eigen::Vector2d> gmmstats::Gaussian2D::sample(const int num_samples) const {
    std::vector<Eigen::Vector2d> samples(num_samples);
    for(int i = 0; i < num_samples; i++) {
        samples[i] = sample();
    }
    return samples;
}
