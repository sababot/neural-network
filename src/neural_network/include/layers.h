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

	Eigen::MatrixXd weight_cache;
	Eigen::VectorXd bias_cache;

	//// Constructor ////
	dense_layer(int, int);

	//// Functions ////
	void forward(Eigen::MatrixXd);
	void backward(Eigen::MatrixXd);
};

class layer_single
{
public:
	layer_single(int, int);

	int n_neurons;
	int n_inputs;

	Eigen::VectorXd biases_single;
	Eigen::VectorXd inputs_single;
	Eigen::MatrixXd weights_single;
	Eigen::VectorXd outputs_single;

	void forward_single(Eigen::VectorXd);
};