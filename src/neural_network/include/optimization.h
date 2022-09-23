#pragma once

#include "../include/layers.h"

#include <eigen3/Eigen/Dense>

class optimizer_sgd
{
public:
	optimizer_sgd(double, double);

	double learning_rate;
	double current_learning_rate;
	double decay;
	int iterations;

	void pre_update_params();
	void update_params(dense_layer *layer);
	void post_update_params();
};