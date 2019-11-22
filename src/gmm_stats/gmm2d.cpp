#include <random>

#include "gmm_stats/gmm2d.h"


gmmstats::GMM2D::GMM2D(const gmmstats::Matrix2Xd & means,
                       const gmmstats::Matrix22Xd & covariances,
                       const std::vector<double> & weights)
{

    const int num_modes = means.size();
    assert(num_modes == covariances.size());
    assert(num_modes == weights.size());

    _gaussians.resize(num_modes);
    _weights.resize(num_modes);

    const double weights_norm = std::accumulate(weights.begin(), weights.end(), 0);  // normalize weight vector
    for(int i = 0; i < num_modes; ++i)
    {
        _gaussians[i] = Gaussian2D(means[i], covariances[i]);
        _weights[i] = weights[i] / weights_norm;
    }
}


double gmmstats::GMM2D::pdfAt(const Eigen::Vector2d & position) const
{
    double weighted_pdf = 0.0;
    for(int i = 0; i < _gaussians.size(); ++i)
    {
        weighted_pdf += _gaussians[i].pdfAt(position) * _weights[i];
    }
    return weighted_pdf;
}


std::vector<double> gmmstats::GMM2D::pdfAt(const std::vector<Eigen::Vector2d> & positions) const
{
    std::vector<double> pdfs(positions.size());
    for(int i = 0; i < positions.size(); ++i)
    {
        pdfs[i] = pdfAt(positions[i]);
    }
    return pdfs;
}


Eigen::Vector2d gmmstats::GMM2D::sample() const
{
    std::discrete_distribution<int> dist(std::begin(_weights), std::end(_weights));
    std::mt19937 gen;
    const int random_mode_id = dist(gen);
    return _gaussians[random_mode_id].sample(); 
}


std::vector<Eigen::Vector2d> gmmstats::GMM2D::sample(const int num_samples) const
{
    std::vector<Eigen::Vector2d> samples(num_samples);
    for(int i = 0; i < num_samples; ++i)
    {
        samples[i] = sample();
    }
    return samples;
}


gmmstats::GMM2D& gmmstats::GMM2D::operator+(const GMM2D& other)
{
    _gaussians.insert(_gaussians.end(), other.gaussians().begin(), other.gaussians().end());
    _weights.insert(_weights.end(), other.weights().begin(), other.weights().end());
    return *this;
}


gmmstats::Gaussian2D gmmstats::GMM2D::mode(const int mode_id) const
{
    assert(0 <= mode_id < _gaussians.size());
    return _gaussians[mode_id];
}


gmmstats::Matrix2Xd gmmstats::GMM2D::mus() const
{
    gmmstats::Matrix2Xd mus(_gaussians.size());
    for(int i = 0; i < _gaussians.size(); ++i)
    {
        mus[i] = _gaussians[i].mean();
    }
    return mus;
}


gmmstats::Matrix22Xd gmmstats::GMM2D::covariances() const
{
    gmmstats::Matrix22Xd covariances;
    for(int i = 0; i < _gaussians.size(); ++i)
    {
        covariances[i] = _gaussians[i].covariance();
    }
    return covariances;
}
