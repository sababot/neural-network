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

Eigen::MatrixXd diagflat(Eigen::VectorXd input)
{
    Eigen::MatrixXd diagflat_matrix(input.rows(), input.rows());
    diagflat_matrix.setZero();

    for (int i = 0; i < diagflat_matrix.rows(); i++)
    {
        diagflat_matrix(i, i) = input(i);
    }

    return diagflat_matrix;
}