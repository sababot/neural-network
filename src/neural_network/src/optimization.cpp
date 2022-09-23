#include "../include/optimization.h"
#include "../include/layers.h"

#include <iostream>

#include <eigen3/Eigen/Eigen>

optimizer_sgd::optimizer_sgd(double learning_rate, double decay)
{
	optimizer_sgd::learning_rate = learning_rate;
	optimizer_sgd::current_learning_rate = learning_rate;
	optimizer_sgd::decay = decay;
	optimizer_sgd::iterations = 0;
}

void optimizer_sgd::pre_update_params()
{
	current_learning_rate = learning_rate * (1.0 / (1 + decay * iterations));
}

void optimizer_sgd::update_params(dense_layer *layer)
{
	layer->weights += (-current_learning_rate * layer->dweights);
	layer->biases += (-current_learning_rate * layer->dbiases);
}

void optimizer_sgd::post_update_params()
{
	iterations++;
}