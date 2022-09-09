#pragma once

#include <eigen3/Eigen/Dense>

class dense_layer
{
private:
	Eigen::VectorXd biases;

public:
	// Output Matrix
	Eigen::MatrixXd weights;
	Eigen::MatrixXd outputs;

	// Constructor
	dense_layer(int, int);

	// Functions
	void forward(Eigen::MatrixXd);
};