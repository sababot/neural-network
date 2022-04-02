#include "../include/neural_network.h"

#include <eigen3/Eigen/Eigen>

double calc_output(Eigen::VectorXd inputs, Eigen::VectorXd weights, double bias)
{
	return inputs(0)*weights(0) + inputs(1)*weights(1) + inputs(2)*weights(2) + bias;
}