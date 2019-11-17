#include <iostream>

#include <eigen3/Eigen/Dense>
#include "gtest/gtest.h"

#include "gmm_stats/gaussian2d.h"


TEST(test_gaussian2d, pdf_at) {
    Eigen::MatrixXd sigma(2, 2);
    sigma << 1, 0.1,
             0.1, 1;
    Eigen::VectorXd mean(2);
    mean << 0, 0;
    Gaussian2D distribution = Gaussian2D(mean, sigma);

    Eigen::Vector2d test;
    test << 0, 0;
    EXPECT_NEAR(distribution.pdf_at(test), 0.16, 1e-4);

    test << -0.6, -0.6;
    EXPECT_NEAR(distribution.pdf_at(test), 0.1153, 1e-4);
}

TEST(test_gaussian2d, sampling)
{
    // Define the covariance matrix and the mean.
    Eigen::Matrix2d sigma(2, 2);
    sigma << 10, 7,
              7, 5;
    Eigen::Vector2d mean(2);
    mean << 2, 2;
    Gaussian2D mvn(mean, sigma);

    // Sample a number of points.
    const unsigned int points = 1000;
    Eigen::MatrixXd x(2, points);
    Eigen::VectorXd vector(2);
    for (unsigned i = 0; i < points; i++)
    {
        vector = mvn.sample(200);
        x(0, i) = vector(0);
        x(1, i) = vector(1);
    }

    // Calculate the mean and convariance of the produces sampled points.
    Eigen::VectorXd approx_mean(2);
    Eigen::MatrixXd approx_sigma(2, 2);
    approx_mean.setZero();
    approx_sigma.setZero();

    for (unsigned int i = 0; i < points; i++)
    {
        approx_mean  = approx_mean  + x.col(i);
        approx_sigma = approx_sigma + x.col(i) * x.col(i).transpose();
    }

    approx_mean  = approx_mean  / static_cast<double>(points);
    approx_sigma = approx_sigma / static_cast<double>(points);
    approx_sigma = approx_sigma - approx_mean * approx_mean.transpose();

    // Check if the statistics of the sampled points are close to the statistics
    // of the given distribution.
    EXPECT_TRUE(approx_mean.isApprox(mean, 5e-1));
    EXPECT_TRUE(approx_sigma.isApprox(sigma, 5e-1));
}
