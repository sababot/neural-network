#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>

Eigen::MatrixXd read_data(std::string, int, int);
Eigen::VectorXd read_data_single(std::string, int);
