#pragma once

#include <eigen3/Eigen/Dense>

class loss_categoral_cross_entropy
{
public:
	double mean_loss;

	Eigen::MatrixXd dinputs;

	void calculate(Eigen::MatrixXd, Eigen::VectorXd);

	Eigen::VectorXd forward(Eigen::MatrixXd, Eigen::VectorXd);
	void backward(Eigen::MatrixXd, Eigen::VectorXd);
};