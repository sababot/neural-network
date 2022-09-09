#include "../include/layers.h"

#include <eigen3/Eigen/Eigen>

// DENSE LAYER
dense_layer::dense_layer(int n_inputs, int n_neurons)
{
	weights = Eigen::MatrixXd::Random(n_inputs, n_neurons) * 0.1;
	biases.resize(n_neurons);
	biases.setZero();
}

void dense_layer::forward(Eigen::MatrixXd inputs)
{
	outputs = inputs * weights;
	outputs += biases.replicate(1, outputs.rows()).transpose();
}