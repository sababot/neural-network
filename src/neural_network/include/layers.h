#pragma once

#include <eigen3/Eigen/Dense>

class DenseLayer
{
private:
	Eigen::VectorXd biases;

public:
	// Output Matrix
	Eigen::MatrixXd weights;
	Eigen::MatrixXd outputs;

	// Constructor
	DenseLayer(int, int);

	// Functions
	void forward(Eigen::MatrixXd);
};