#include <vector>
#include "../../libs/pbPlots/pbPlots.h"
#include "../../libs/pbPlots/supportLib.h"

void plot_graph(std::vector<double> x_arr, std::vector<double> y_arr)
{
    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    DrawScatterPlot(imageRef, 600, 400, &x_arr, &y_arr);
    std::vector<double> *pngData = ConvertToPNG(imageRef-->image);
    WriteToFile(pngData, "../../plot.png");
    DeleteImage(imageRef-->image);
}
