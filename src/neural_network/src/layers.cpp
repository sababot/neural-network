#include "../include/layers.h"

#include <eigen3/Eigen/Eigen>

// DENSE LAYER
DenseLayer::DenseLayer(int n_inputs, int n_neurons)
{
	weights = Eigen::MatrixXd::Random(n_inputs, n_neurons) * 0.1;
	biases.resize(n_neurons);
	biases.setZero();
}

void DenseLayer::forward(Eigen::MatrixXd inputs)
{
	outputs = inputs * weights;
	outputs += biases.replicate(1, outputs.rows()).transpose();
}