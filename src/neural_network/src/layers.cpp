#include "../include/layers.h"

#include <eigen3/Eigen/Eigen>
#include <iostream>

// DENSE LAYER
dense_layer::dense_layer(int n_inputs, int n_neurons)
{
	dense_layer::n_inputs = n_inputs;
	dense_layer::n_neurons = n_neurons;
	
	weights = Eigen::MatrixXd::Random(dense_layer::n_inputs, dense_layer::n_neurons) * 0.01;
	biases.resize(dense_layer::n_neurons);
	biases.setZero();
}

void dense_layer::forward(Eigen::MatrixXd inputs)
{
	dense_layer::inputs = inputs;

	outputs = (inputs * weights);
	outputs += biases.replicate(1, outputs.rows()).transpose();
}

void dense_layer::backward(Eigen::MatrixXd dvalues)
{
	dweights = inputs.transpose() * dvalues;
	
	dbiases.resize(dvalues.cols());
	dbiases.setZero();
	for (int i = 0; i < dvalues.cols(); i++)
	{
		for (int j = 0; j < dvalues.rows(); j++)
		{
			dbiases(i) += dvalues(j, i);
		}
	}

	dinputs = dvalues * weights.transpose();
}