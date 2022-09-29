#include "../include/optimization.h"
#include "../include/layers.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Eigen>

/********** Stochastic Gradient Descent **********/
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

/********** Adaptive Gradient **********/
optimizer_adagrad::optimizer_adagrad(double learning_rate, double decay, double epsilon)
{
	optimizer_adagrad::learning_rate = learning_rate;
	optimizer_adagrad::current_learning_rate = learning_rate;
	optimizer_adagrad::decay = decay;
	optimizer_adagrad::iterations = 0;
	optimizer_adagrad::epsilon = epsilon;
}

void optimizer_adagrad::pre_update_params()
{
	current_learning_rate = learning_rate * (1.0 / (1 + decay * iterations));
}

void optimizer_adagrad::update_params(dense_layer *layer)
{
	if (layer->weight_cache.rows() == 0 || layer->bias_cache.rows() == 0)
	{
		layer->weight_cache.resize(layer->weights.rows(), layer->weights.cols());
		layer->weight_cache.setZero();
		layer->bias_cache.resize(layer->biases.rows(), layer->biases.cols());
		layer->bias_cache.setZero();
	}

	layer->weight_cache = layer->weight_cache.array() + layer->dweights.array().pow(2);
	layer->bias_cache = layer->bias_cache.array() + layer->dbiases.array().pow(2);

	Eigen::MatrixXd tmp1 = layer->weight_cache.array().sqrt() + epsilon;
	layer->weights = layer->weights.array() + (-current_learning_rate * layer->dweights.array() * tmp1.array().inverse());
	
	Eigen::MatrixXd tmp2 = layer->bias_cache.array().sqrt() + epsilon;
	layer->biases = layer->biases.array() + (-current_learning_rate * layer->dbiases.array() * tmp2.array().inverse());
}

void optimizer_adagrad::post_update_params()
{
	iterations++;
}