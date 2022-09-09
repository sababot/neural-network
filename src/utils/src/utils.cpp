#include <vector>
#include <eigen3/Eigen/Dense>

Eigen::MatrixXd convert_to_onehot(Eigen::VectorXd input, int n_classes)
{
    Eigen::MatrixXd onehot_matrix(input.rows(), n_classes);

    onehot_matrix.setZero();

    for (int i = 0; i < onehot_matrix.rows(); i++)
    {
        int index = input(i);
        onehot_matrix(i, index) = 1;
    }

    return onehot_matrix;
}