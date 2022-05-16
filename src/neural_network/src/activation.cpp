#include "../include/activation.h"

#include <eigen3/Eigen/Eigen>

Eigen::MatrixXd ActivationReLU(Eigen::MatrixXd inputs)
{
	return inputs.array().max(0);
}