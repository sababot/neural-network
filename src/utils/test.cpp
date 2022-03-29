#include "include/utils.h"

#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace std;

int main()
{
    Eigen::MatrixXd mat(5, 1);
    mat <<  3, 3, 2, 2, 1;

    mat = class_to_output(mat, 3);
    std::cout << mat << endl;

    return 0;
}
