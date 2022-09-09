#include <vector>
#include <eigen3/Eigen/Dense>

Eigen::MatrixXd convert_to_onehot(Eigen::VectorXd input, int n_classes)
{
    Eigen::MatrixXd onehot_matrix(input.rows(), n_classes);

    onehot_matrix.setZero();

    return onehot_matrix;
}