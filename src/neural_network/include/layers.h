#pragma once

#include <eigen3/Eigen/Dense>

class dense_layer
{
public:
	// Size
	int n_neurons;
	int n_inputs;

	// Output Matrix
	Eigen::VectorXd biases;
	Eigen::MatrixXd weights;
	Eigen::MatrixXd outputs;

	// Constructor
	dense_layer(int, int);

	// Functions
	void forward(Eigen::MatrixXd);
};