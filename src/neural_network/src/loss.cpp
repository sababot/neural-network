#include "../include/loss.h"

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <cmath>

using namespace std;

double calculate_loss(Eigen::MatrixXd softmax_outputs, Eigen::VectorXd class_targets)
{
	Eigen::VectorXd out(softmax_outputs.rows());

	for (int i = 0; i < softmax_outputs.rows(); i++)
	{
		int class_target_index = class_targets(i);
		out(i) = -log(softmax_outputs(i, class_target_index));
	}

	double mean_loss;

	for (int i = 0; i < out.rows(); i++)
	{
		mean_loss += out(i);
	}

	mean_loss /= out.rows();

	return mean_loss;
}