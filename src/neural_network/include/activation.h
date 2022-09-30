#pragma once

#include <eigen3/Eigen/Dense>

class activation_relu
{
public:
	Eigen::MatrixXd inputs;
	Eigen::MatrixXd outputs;
	Eigen::MatrixXd dinputs;

	Eigen::VectorXd outputs_single;

	void forward(Eigen::MatrixXd);
	void forward_single(Eigen::VectorXd);
	void backward(Eigen::MatrixXd);
};

class activation_softmax
{
public:
	Eigen::MatrixXd inputs;
	Eigen::MatrixXd outputs;
	Eigen::MatrixXd dinputs;

	Eigen::VectorXd outputs_single;

	void forward(Eigen::MatrixXd);
	void forward_single(Eigen::VectorXd);
	void backward(Eigen::MatrixXd);
};