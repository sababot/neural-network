#pragma once

#include <eigen3/Eigen/Eigen>

Eigen::MatrixXd convert_to_onehot(Eigen::VectorXd, int);
Eigen::MatrixXd diagflat(Eigen::VectorXd);
