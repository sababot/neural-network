#include "../include/loss.h"
#include "../../utils/include/utils.h"

#include <eigen3/Eigen/Eigen>
#include <cmath>
#include <iostream>

using namespace std;

/********** Categoral Cross Entropy **********/
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
	Eigen::MatrixXd clip (2, 1);
	clip << 0.0000001, // min
    		  0.9999999; // max

	Eigen::MatrixXd softmax_outputs_predicted = softmax_outputs.cwiseMin(clip.replicate(1, softmax_outputs.cols()).row(1).colwise().replicate(softmax_outputs.rows()))
         .cwiseMax(clip.replicate(1, softmax_outputs.cols()).row(0).colwise().replicate(softmax_outputs.rows()));

	Eigen::VectorXd out(softmax_outputs_predicted.rows());

	for (int i = 0; i < softmax_outputs_predicted.rows(); i++)
	{
		int class_target_index = class_targets(i);
		out(i) = -log(softmax_outputs_predicted(i, class_target_index));
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

/********** Softmax + Categoral Cross Entropy **********/
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