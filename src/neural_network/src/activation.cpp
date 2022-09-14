#include "../include/activation.h"

#include <eigen3/Eigen/Eigen>
#include <cmath>

using namespace std;

void activation_relu::forward(Eigen::MatrixXd inputs)
{
	activation_relu::inputs = inputs;
	outputs = inputs.array().max(0);
}

void activation_relu::backward(Eigen::MatrixXd dvalues)
{
	dinputs = dvalues;

	dinputs = (inputs.array() <= 0).select(0, dinputs);
}

void activation_softmax::forward(Eigen::MatrixXd inputs)
{
	activation_softmax::inputs = inputs;

	// unormalized probabilities
	Eigen::MatrixXd exp_values(inputs.rows(), inputs.cols());

	for (int i = 0; i < exp_values.rows(); i++)
	{
		for (int j = 0; j < exp_values.cols(); j++)
			exp_values(i, j) = exp(inputs(i, j) - inputs.row(i).maxCoeff());
	}

	// normalized probabilities
	Eigen::MatrixXd norm_values(inputs.rows(), inputs.cols());

	for (int i = 0; i < norm_values.rows(); i++)
	{
		double sum = exp_values.row(i).sum();

		for (int j = 0; j < norm_values.cols(); j++)
		{
			norm_values(i, j) = exp_values(i, j) / sum;
		}
	}

	outputs = norm_values;
}