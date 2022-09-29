#include "../include/activation.h"
#include "../../utils/include/utils.h"

#include <eigen3/Eigen/Eigen>
#include <cmath>
#include <iostream>

using namespace std;

/********** ReLU **********/
void activation_relu::forward(Eigen::MatrixXd inputs)
{
	activation_relu::inputs = inputs;
	outputs = inputs.array().max(0);
}

void activation_relu::backward(Eigen::MatrixXd dvalues)
{
	dinputs = dvalues;

	for (int i = 0; i < inputs.rows(); i++)
	{
		for (int j = 0; j < inputs.cols(); j++)
		{
			if (inputs(i, j) <= 0)
			{
				dinputs(i, j) = 0;
			}
		}
	}
}

/********** Softmax **********/
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

void activation_softmax::backward(Eigen::MatrixXd dvalues)
{
	dinputs.resize(dvalues.rows(), dvalues.cols());

	for (int i = 0; i < outputs.rows(); i++)
	{
		Eigen::MatrixXd single_output(outputs.cols(), 1);
		for (int j = 0; j < single_output.rows(); j++)
			single_output(j, 0) = outputs(i, j);

		Eigen::VectorXd single_dvalues = dvalues.row(i);

		Eigen::MatrixXd jacobian_matrix = diagflat(outputs.row(i)) - (single_output * single_output.transpose());

		dinputs.row(i) = jacobian_matrix * single_dvalues;
	}
}