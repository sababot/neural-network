#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "../include/createdata.h"

void create_data(int points, int classes)
{
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    X.resize(points * classes, 2);
    X.setZero();

    y.resize(points * classes);
    y.setZero();

    for (int i = 0; i < classes; i++)
    {
        Eigen::VectorXd ix, r, t;
        ix << Eigen::VectorXd::LinSpaced(((points * (i + 1)) - (points * i)), points * i, points * (i + 1));
        r = Eigen::VectorXd::LinSpaced(0.0, 1, points);
        t = Eigen::VectorXd::LinSpaced(i * 4, (i + 1) * 4, points) + Eigen::VectorXd::Random(points) * 0.2;
        
        for (int j = ((points * (i + 1)) - (points * i)); j < points * i, points * (i + 1); j++)
        {
            X(j) = 
        }

        y(ix) = i;
    }
}

/*
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
*/