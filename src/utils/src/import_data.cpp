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

double read_double(string filename)
{
    double input;

    // file pointer
    fstream fin;

    string line;

    // open file
    fin.open(filename);
    fin >> line;

    stringstream  ss(line);
    string str;
    getline(ss, str, ',');
    input = stod(str);

    return input;
}

void export_data(Eigen::MatrixXd data, string name)
{
    ofstream myfile;
    myfile.open (name);

    for (int i = 0; i < data.rows(); i++)
    {
        string row_output = "";

        for (int j = 0; j < data.cols(); j++)
        {
            if (j != data.cols() - 1)
                row_output += to_string(data(i, j)) + ",";
            else
                row_output += to_string(data(i, j));
        }

        myfile << row_output + "\n";
    }

    myfile.close();
}