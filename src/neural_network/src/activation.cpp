#include "../include/activation.h"

#include <eigen3/Eigen/Eigen>
#include <cmath>

using namespace std;

Eigen::MatrixXd ActivationReLU(Eigen::MatrixXd inputs)
{
	return inputs.array().max(0);
}

Eigen::MatrixXd ActivationSoftmax(Eigen::MatrixXd inputs)
{
	Eigen::MatrixXd exp_values(inputs.rows(), inputs.cols());

	for (int i = 0; i < exp_values.rows(); i++)
	{
		for (int j = 0; j < exp_values.cols(); j++)
			exp_values(i, j) = exp(inputs(i, j) - inputs.maxCoeff());
	}

	Eigen::MatrixXd norm_values(inputs.rows(), inputs.cols());

	for (int i = 0; i < norm_values.rows(); i++)
	{
		double sum = exp_values.row(i).sum();

		for (int j = 0; j < norm_values.cols(); j++)
		{
			norm_values(i, j) = exp_values(i, j) / sum;
		}
	}

	return norm_values;
}

double calculate_loss(double prediction, double target)
{

}