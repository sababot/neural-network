#include "../include/layers.h"

#include <eigen3/Eigen/Eigen>

// DENSE LAYER
dense_layer::dense_layer(int x, int y)
{
	n_neurons = y;
	n_inputs = x;
	
	weights = Eigen::MatrixXd::Random(n_inputs, n_neurons) * 0.05;
	biases.resize(n_neurons);
	biases.setZero();
}

void dense_layer::forward(Eigen::MatrixXd inputs)
{
	outputs = inputs * weights;
	outputs += biases.replicate(1, outputs.rows()).transpose();
}