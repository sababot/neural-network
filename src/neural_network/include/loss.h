#pragma once

#include <eigen3/Eigen/Dense>
#include "activation.h"

class loss_categoral_cross_entropy
{
public:
	double mean_loss;

	Eigen::MatrixXd dinputs;
	Eigen::VectorXd outputs;
	double accumulated_sum;
	double accumulated_count;

	void calculate(Eigen::MatrixXd, Eigen::VectorXd);
	double calculate_accumulated();
	void new_pass();

	Eigen::VectorXd forward(Eigen::MatrixXd, Eigen::VectorXd);
	void backward(Eigen::MatrixXd, Eigen::VectorXd);
};

class activation_softmax_loss_categoral_cross_entropy
{
public:
	activation_softmax activation;
	loss_categoral_cross_entropy loss;

	Eigen::MatrixXd outputs;
	Eigen::MatrixXd dinputs;


	double forward(Eigen::MatrixXd, Eigen::VectorXd);
	void backward(Eigen::MatrixXd, Eigen::VectorXd);
};
