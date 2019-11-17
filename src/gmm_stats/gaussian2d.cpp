#include <cassert>
#include <cmath>

#include "gmm_stats/gaussian2d.h"


Gaussian2D::Gaussian2D(const Eigen::Vector2d & mean, const Eigen::Matrix2d & covariance) {
    assert(covariance.determinant() != 0);
    assert(covariance(0, 1) == covariance(1, 0));

    mu = mean;
    sigma = covariance;

    _K1 = 1 / (2 * M_PI * sqrt(sigma.determinant()));
    _K2 = -0.5 / (sigma(0, 0) * sigma(1, 1) - pow(sigma(0, 1), 2));
}


double Gaussian2D::pdf_at(const Eigen::Vector2d & position) const {
    double sqrt2pi = std::sqrt(2 * M_PI);
    double quadform  = (position - mu).transpose() * sigma.inverse() * (position - mu);
    double norm = std::pow(sqrt2pi, - 2) *
                  std::pow(sigma.determinant(), - 0.5);
    return norm * exp(-0.5 * quadform);
}


std::vector<double> Gaussian2D::pdf_at(const std::vector<Eigen::Vector2d> & positions) const {
    std::vector<double> pdfs(positions.size());
    for(int i = 0; i < positions.size(); i++) {
        pdfs[i] = pdf_at(positions[i]);
    }
    return pdfs;
}


Eigen::Vector2d Gaussian2D::sample(int num_iterations) const {
    // Generate x from the N(0, I) distribution
    Eigen::Vector2d x;
    Eigen::Vector2d sum;
    sum.setZero();
    for(unsigned int i = 0; i < num_iterations; i++) {
        x.setRandom();
        x = 0.5 * (x + Eigen::VectorXd::Ones(2));
        sum = sum + x;
    }
    sum = sum - (static_cast<double>(num_iterations) / 2) * Eigen::VectorXd::Ones(2);
    x = sum / (std::sqrt(static_cast<double>(num_iterations) / 12));

    // Find the eigen vectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(sigma);
    Eigen::MatrixXd eigenvectors = eigen_solver.eigenvectors().real();

    // Find the eigenvalues of the covariance matrix
    Eigen::MatrixXd eigenvalues = eigen_solver.eigenvalues().real().asDiagonal();

    // Find the transformation matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigenvalues);
    Eigen::MatrixXd sqrt_eigenvalues = es.operatorSqrt();
    Eigen::MatrixXd Q = eigenvectors * sqrt_eigenvalues;

    return Q * x + mu;
}


std::vector<Eigen::Vector2d> Gaussian2D::sample_vector(const int num_samples) const {
    std::vector<Eigen::Vector2d> samples(num_samples);
    for(int i = 0; i < num_samples; i++) {
        samples[i] = sample();
    }
    return samples;
}