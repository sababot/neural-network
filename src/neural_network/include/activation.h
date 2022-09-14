#pragma once

#include <eigen3/Eigen/Dense>

class activation_relu
{
public:
	Eigen::MatrixXd inputs;
	Eigen::MatrixXd outputs;
	Eigen::MatrixXd dinputs;

	void forward(Eigen::MatrixXd);
	void backward(Eigen::MatrixXd);
};

class activation_softmax
{
public:
	Eigen::MatrixXd inputs;
	Eigen::MatrixXd outputs;
	Eigen::MatrixXd dinputs;

	void forward(Eigen::MatrixXd);
	void backward(Eigen::MatrixXd);
};