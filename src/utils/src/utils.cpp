#include <vector>
#include <eigen3/Eigen/Dense>

#include "../../../libs/pbPlots/pbPlots.h"
#include "../../../libs/pbPlots/supportLib.h"

void plot_graph(std::vector<double> x_arr, std::vector<double> y_arr)
{
    StringReference *errorMessage = new StringReference();
    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    DrawScatterPlot(imageRef, 600, 400, &x_arr, &y_arr, errorMessage);
    std::vector<double> *pngData = ConvertToPNG(imageRef->image);
    WriteToFile(pngData, "plot.png");
    DeleteImage(imageRef->image);
}

Eigen::MatrixXd output_to_class(Eigen::MatrixXd m, int max)
{
    Eigen::MatrixXd output(m.rows(), 1);

    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < max; j++)
        {
            if (m(i, j) == 1)
            {
                output(i, 0) = j + 1;
            }
        }
    }

    return output;
}

Eigen::MatrixXd class_to_output(Eigen::MatrixXd m, int max)
{
    Eigen::MatrixXd output(m.rows(), max);

    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < max; j++)
        {
            if (j == m(i, 0) - 1)
            {
                output(i, j) = 1;
            }

            else
            {
                output(i, j) = 0;
            }
        }
    }

    return output;
}
