#include <eigen3/Eigen/Dense>
#include "gtest/gtest.h"

#include "gmm_stats/gaussian2d.h"
#include "gmm_stats/gmm2d.h"


TEST(test_gaussian2d, pdf_at)
{
    Eigen::MatrixXd sigma(2, 2);
    sigma << 1, 0.1,
             0.1, 1;
    Eigen::VectorXd mean(2);
    mean << 0, 0;
    gmmstats::Gaussian2D distribution(mean, sigma);

    Eigen::Vector2d test;
    test << 0, 0;
    EXPECT_NEAR(distribution.pdfAt(test), 0.16, 1e-4);

    test << -0.6, -0.6;
    EXPECT_NEAR(distribution.pdfAt(test), 0.1153, 1e-4);
}


TEST(test_gaussian2d, sampling)
{
    // Define the covariance matrix and the mean.
    Eigen::Matrix2d sigma(2, 2);
    sigma << 10, 7,
              7, 5;
    Eigen::Vector2d mean(2);
    mean << 2, 2;
    gmmstats::Gaussian2D mvn(mean, sigma);

    // Sample a number of points.
    const unsigned int points = 1000;
    Eigen::MatrixXd x(2, points);
    Eigen::VectorXd vector(2);
    for(unsigned i = 0; i < points; i++) {
        vector = mvn.sample();
        x(0, i) = vector(0);
        x(1, i) = vector(1);
    }

    // Calculate the mean and convariance of the produces sampled points.
    Eigen::VectorXd approx_mean(2);
    Eigen::MatrixXd approx_sigma(2, 2);
    approx_mean.setZero();
    approx_sigma.setZero();

    for(unsigned int i = 0; i < points; i++) {
        approx_mean  += x.col(i);
        approx_sigma += x.col(i) * x.col(i).transpose();
    }

    approx_mean  = approx_mean  / static_cast<double>(points);
    approx_sigma = approx_sigma / static_cast<double>(points);
    approx_sigma = approx_sigma - approx_mean * approx_mean.transpose();

    // Check if the statistics of the sampled points are close to the statistics of the given distribution.
    EXPECT_TRUE(approx_mean.isApprox(mean, 5e-1));
    EXPECT_TRUE(approx_sigma.isApprox(sigma, 5e-1));
}


TEST(test_gmm2d, pdf_at)
{
    gmmstats::Matrix2Xd means(2);
    means[0] << 0, 0;
    means[1] << 0, 0;
    gmmstats::Matrix22Xd covariances(2);
    covariances[0] << 1, 0.1, 0.1, 1;
    covariances[1] << 1, 0.1, 0.1, 1;
    std::vector<double> weights(2);
    weights[0] = 1.0;
    weights[1] = 1.0;
    gmmstats::GMM2D gmm(means, covariances, weights);

    Eigen::Vector2d test;
    test << 0, 0;
    EXPECT_NEAR(gmm.pdfAt(test), 0.16, 1e-4);

    test << -0.6, -0.6;
    EXPECT_NEAR(gmm.pdfAt(test), 0.1153, 1e-4);
}


TEST(test_gmm2d, sampling)
{
    gmmstats::Matrix2Xd means(2);
    means[0] << 10, 0;
    means[1] << 0, 10;
    gmmstats::Matrix22Xd covariances(2);
    covariances[0] << 0.01, 0, 0, 0.01;
    covariances[1] << 0.01, 0, 0, 0.01;
    std::vector<double> weights(2);
    weights[0] = 10.0;
    weights[1] = 1.0;
    gmmstats::GMM2D gmm(means, covariances, weights);

    const int num_samples = 100;
    std::vector<Eigen::Vector2d> samples = gmm.sample(num_samples);

    Eigen::Vector2d approx_mean;
    approx_mean.setZero();
    for (auto & element : samples) {
        approx_mean += element;
    }
    approx_mean = approx_mean / static_cast<double>(num_samples);

    // Check if the statistics of the sampled points are close to the statistics of the given distribution.
    EXPECT_NEAR(approx_mean(0), 10, 0.1);
    EXPECT_NEAR(approx_mean(1), 0, 0.1);
}