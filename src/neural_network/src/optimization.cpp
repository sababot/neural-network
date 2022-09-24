#include "../include/optimization.h"
#include "../include/layers.h"

#include <iostream>

#include <eigen3/Eigen/Eigen>

optimizer_sgd::optimizer_sgd(double learning_rate, double decay, double momentum)
{
	optimizer_sgd::learning_rate = learning_rate;
	optimizer_sgd::current_learning_rate = learning_rate;
	
	optimizer_sgd::decay = decay;
	optimizer_sgd::momentum = momentum;

	optimizer_sgd::iterations = 0;
}

void optimizer_sgd::pre_update_params()
{
	current_learning_rate = learning_rate * (1.0 / (1 + decay * iterations));
}

void optimizer_sgd::update_params(dense_layer *layer)
{
	if (layer->weight_momentums.rows() == 0 || layer->bias_momentums.rows() == 0)
	{
		layer->weight_momentums.resize(layer->weights.rows(), layer->weights.cols());
		layer->weight_momentums.setZero();

		layer->bias_momentums.resize(layer->biases.rows());
		layer->bias_momentums.setZero();
	}

	weight_updates = momentum * layer->weight_momentums - current_learning_rate * layer->dweights;
	layer->weight_momentums = weight_updates;

	bias_updates = momentum * layer->bias_momentums - current_learning_rate * layer->dbiases;
	layer->bias_momentums = bias_updates;

	layer->weights += weight_updates;
	layer->biases += bias_updates;
}

void optimizer_sgd::post_update_params()
{
	iterations++;
}