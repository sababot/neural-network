#pragma once

#include "../include/layers.h"

#include <eigen3/Eigen/Dense>

class optimizer_sgd
{
public:
	optimizer_sgd(double);

	double learning_rate;
	void update_params(dense_layer *layer);
};