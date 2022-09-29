#pragma once

#include "../include/layers.h"

#include <eigen3/Eigen/Dense>

class optimizer_sgd
{
public:
	optimizer_sgd(double, double, double);

	double learning_rate;
	double current_learning_rate;
	double decay;
	double momentum;
	int iterations;

	Eigen::MatrixXd weight_updates;
	Eigen::VectorXd bias_updates;

	void pre_update_params();
	void update_params(dense_layer *layer);
	void post_update_params();
};

class optimizer_adagrad
{
public:
	optimizer_adagrad(double, double, double);

	double learning_rate;
	double current_learning_rate;
	double decay;
	double epsilon;
	int iterations;

	Eigen::MatrixXd weight_updates;
	Eigen::VectorXd bias_updates;

	void pre_update_params();
	void update_params(dense_layer *layer);
	void post_update_params();
};

class optimizer_rmsprop
{
public:
	optimizer_rmsprop(double, double, double, double);

	double learning_rate;
	double current_learning_rate;
	double decay;
	double epsilon;
	double rho;
	int iterations;

	Eigen::MatrixXd weight_updates;
	Eigen::VectorXd bias_updates;

	void pre_update_params();
	void update_params(dense_layer *layer);
	void post_update_params();
};