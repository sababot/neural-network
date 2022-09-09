#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>

#include "../include/import_data.h"

using namespace std;

Eigen::MatrixXd read_data(string filename, int n_rows, int n_cols)
{
    Eigen::MatrixXd inputs(n_rows, n_cols);

    // file pointer
    fstream fin;

    string line;

    // open file
    fin.open(filename);

    for (int i = 0; i < n_rows; i++)
    {
        fin >> line;

        stringstream  ss(line);
        string str;
        for (int j = 0; j < n_cols; j++)
        {
            getline(ss, str, ',');
            inputs(i, j) = stod(str);
        }
    }

    return inputs;
}

Eigen::VectorXd read_data_single(string filename, int n_rows)
{
    Eigen::VectorXd inputs(n_rows);

    // file pointer
    fstream fin;

    string line;

    // open file
    fin.open(filename);

    for (int i = 0; i < n_rows; i++)
    {
        fin >> line;

        stringstream  ss(line);
        string str;
        getline(ss, str, ',');
        inputs(i) = stod(str);
    }

    return inputs;
}