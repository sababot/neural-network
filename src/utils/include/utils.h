#pragma once

#include <eigen3/Eigen/Eigen>

Eigen::MatrixXd convert_to_onehot(Eigen::VectorXd, int);
Eigen::MatrixXd diagflat(Eigen::VectorXd);
Eigen::VectorXd argmax(Eigen::MatrixXd);
double calculate_accuracy(Eigen::MatrixXd, Eigen::VectorXd);
Eigen::MatrixXd split_rows(Eigen::MatrixXd, int, int);
