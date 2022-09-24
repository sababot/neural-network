#pragma once

#include <eigen3/Eigen/Dense>

class dense_layer
{
public:
	//// Variables ////
	// Forward
	int n_neurons;
	int n_inputs;

	Eigen::MatrixXd inputs;

	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;
	Eigen::MatrixXd outputs;

	// Backward
	Eigen::MatrixXd dweights;
	Eigen::VectorXd dbiases;
	Eigen::MatrixXd dinputs;

	// Optimization
	Eigen::MatrixXd weight_momentums;
	Eigen::VectorXd bias_momentums;

	//// Constructor ////
	dense_layer(int, int);

	//// Functions ////
	void forward(Eigen::MatrixXd);
	void backward(Eigen::MatrixXd);
};