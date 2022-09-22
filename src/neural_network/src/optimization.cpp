#include "../include/optimization.h"
#include "../include/layers.h"

#include <iostream>

#include <eigen3/Eigen/Eigen>

optimizer_sgd::optimizer_sgd(double learning_rate)
{
	optimizer_sgd::learning_rate = learning_rate;
}

void optimizer_sgd::update_params(dense_layer *layer)
{
	layer->weights += (-learning_rate * layer->dweights);
	layer->biases += (-learning_rate * layer->dbiases);
}