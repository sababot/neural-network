#include "../include/loss.h"
#include "../../utils/include/utils.h"

#include <eigen3/Eigen/Eigen>
#include <cmath>
#include <iostream>

using namespace std;

void loss_categoral_cross_entropy::calculate(Eigen::MatrixXd softmax_outputs, Eigen::VectorXd class_targets)
{
	// sample losses
	Eigen::VectorXd sample_losses = forward(softmax_outputs, class_targets);

	// mean loss
	for (int i = 0; i < sample_losses.rows(); i++)
	{
		mean_loss += sample_losses(i);
	}

	mean_loss /= sample_losses.rows();
}

Eigen::VectorXd loss_categoral_cross_entropy::forward(Eigen::MatrixXd softmax_outputs, Eigen::VectorXd class_targets)
{
	Eigen::VectorXd out(softmax_outputs.rows());

	for (int i = 0; i < softmax_outputs.rows(); i++)
	{
		int class_target_index = class_targets(i);
		out(i) = -log(softmax_outputs(i, class_target_index));
	}

	outputs = out;

	return out;
}

void loss_categoral_cross_entropy::backward(Eigen::MatrixXd dvalues, Eigen::VectorXd y_true)
{
	int samples = dvalues.rows();
	int labels = dvalues.cols();

	Eigen::MatrixXd y_true_onehot = convert_to_onehot(y_true, labels);

	dinputs = -y_true_onehot * dvalues.inverse();
	dinputs /= samples;
}

///////////////////// ------------------- ////////////////////////
double activation_softmax_loss_categoral_cross_entropy::forward(Eigen::MatrixXd inputs, Eigen::VectorXd y_true)
{
	activation.forward(inputs);
	outputs = activation.outputs;

	loss.calculate(outputs, y_true);
	
	return loss.mean_loss;
}

void activation_softmax_loss_categoral_cross_entropy::backward(Eigen::MatrixXd dvalues, Eigen::VectorXd y_true)
{
	int samples = dvalues.rows();

	// y_true has to be discrete values, not one-hot
	dinputs = dvalues;

	for (int i = 0; i < samples; i++)
	{
		int y = y_true(i);
		dinputs(i, y) -= 1.0;
	}

	dinputs = dinputs / samples;
}