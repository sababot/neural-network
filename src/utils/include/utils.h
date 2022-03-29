#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

void plot_graph(std::vector<double>, std::vector<double>);
Eigen::MatrixXd output_to_class(Eigen::MatrixXd, int);
Eigen::MatrixXd class_to_output(Eigen::MatrixXd, int);
