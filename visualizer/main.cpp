#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

#include "../src/neural_network/include/layers.h"
#include "../src/neural_network/include/activation.h"
#include "../src/utils/include/import_data.h"

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <cmath>
#include <algorithm>

using namespace std;

// init model
layer_single layer1(784, 64);
activation_relu activation1;

layer_single layer2(64, 10);
activation_softmax activation2;

// parameters
Eigen::MatrixXd weights1 = read_data("../model/weights1.csv", 64, 784);
Eigen::MatrixXd weights2 = read_data("../model/weights2.csv", 10, 64);
Eigen::VectorXd bias1 = read_data_single("../model/biases1.csv", 64);
Eigen::VectorXd bias2 = read_data_single("../model/biases2.csv", 10);

void load_model(Eigen::MatrixXd w1, Eigen::MatrixXd w2, Eigen::VectorXd b1, Eigen::VectorXd b2)
{
    layer1.weights_single = w1;
    layer2.weights_single = w2;

    layer1.biases_single = b1;
    layer2.biases_single = b2;
}

Eigen::VectorXd output(Eigen::VectorXd in)
{
    layer1.forward_single(in);
    activation1.forward_single(layer1.outputs_single);
    layer2.forward_single(activation1.outputs_single);
    activation2.forward_single(layer2.outputs_single);

    double pred_tmp = 0;
    double pred;

    for (int i = 0; i < activation2.outputs_single.rows(); i++)
    {
        if (activation2.outputs_single(i) > pred_tmp)
        {
            pred_tmp = activation2.outputs_single(i);
            pred = i;
        }
    }

    return activation2.outputs_single;
}

class Paint : public olc::PixelGameEngine
{
public:
	Paint()
	{
		sAppName = "digit drawer";
	}
    
    int pixel_width = 15;
private:
    std::vector<array<int, 2>> pixels;
    Eigen::VectorXd inputs;
    int x = 0;
    int y = 0;
    int text_scale = 2;
    int confidences[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Eigen::VectorXd predictions;

public:
	bool OnUserCreate() override
	{
        inputs.resize(784);
        inputs.setZero();
        predictions.resize(10);

        predictions = output(inputs);

        Eigen::VectorXd tmp = predictions;
        for (int i = 0; i < tmp.rows(); i++)
        {
            double max = 0;
            int index;
            for (int j = 0; j < tmp.rows(); j++)
            {
                if (tmp(j) > max)
                {
                    max = tmp(j);
                    index = j;
                }
            }

            confidences[index] = i + 1;
            tmp(index) = 0.0;
        }

        return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
        // Erase previous frame
	    Clear({20, 20, 20});

/********** DIGIT DRAWER START **********/
        // Draw
        for (int i = 0; i < pixels.size(); i++)
        {
            if ((int(pixels[i][0]) % pixel_width == 0) && (int(pixels[i][1]) % pixel_width == 0))
                FillRect(int(pixels[i][0]), int(pixels[i][1]), pixel_width, pixel_width, {175, 175, 175});
        }

        // Cursor
        FillRect(int(x), int(y), pixel_width, pixel_width, {175, 175, 175});
        
        if (GetMouseX() <= (28 * pixel_width) + (pixel_width - 1) && GetMouseY() <= (28 * pixel_width) + (pixel_width - 1))
        {
            if ((int(GetMouseX())) % pixel_width != 0)
                x = int(GetMouseX()) - (int(GetMouseX())) % pixel_width;
            else
                x = int(GetMouseX());

            if ((int(GetMouseY())) % pixel_width != 0)
                y = int(GetMouseY()) - (int(GetMouseY())) % pixel_width;
            else
                y = int(GetMouseY());
        }

        // Paint
        if (GetMouse(0).bHeld && GetMouseY() <= (28 * pixel_width) + (pixel_width - 1) && GetMouseX() <= (28 * pixel_width) + (pixel_width - 1))
        {
            pixels.push_back({x, y});

            for (int i = 0; i < inputs.rows(); i++)
            {
                int x2 = i % 28;
                int y2 = floor(i / 28);
                array<int, 2> coord = {(pixel_width * x2) + pixel_width, (pixel_width * y2) + pixel_width};

                if (count(pixels.begin(), pixels.end(), coord))
                    inputs(i) = 225;
                else
                    inputs(i) = 0;
            }

            predictions = output(inputs);

            Eigen::VectorXd tmp = predictions;
            for (int i = 0; i < tmp.rows(); i++)
            {
                double max = 0;
                int index;
                for (int j = 0; j < tmp.rows(); j++)
                {
                    if (tmp(j) > max)
                    {
                        max = tmp(j);
                        index = j;
                    }
                }

                confidences[index] = i + 1;
                tmp(index) = 0.0;
            }
        }
        
        // Erase
        if (GetMouse(1).bHeld && GetMouseY() <= (28 * pixel_width) + (pixel_width - 1) && GetMouseX() <= (28 * pixel_width) + (pixel_width - 1))
        {
            for (int i = 0; i < pixels.size(); i++)
            {
                if (pixels[i][0] == x && pixels[i][1] == y)
                    pixels.erase(pixels.begin() + i);
            }

            for (int i = 0; i < inputs.rows(); i++)
            {
                int x2 = i % 28;
                int y2 = floor(i / 28);
                array<int, 2> coord = {(pixel_width * x2) + pixel_width, (pixel_width * y2) + pixel_width};

                if (count(pixels.begin(), pixels.end(), coord))
                    inputs(i) = 225;
                else
                    inputs(i) = 0;
            }

            predictions = output(inputs);

            Eigen::VectorXd tmp = predictions;
            for (int i = 0; i < tmp.rows(); i++)
            {
                double max = 0;
                int index;
                for (int j = 0; j < tmp.rows(); j++)
                {
                    if (tmp(j) > max)
                    {
                        max = tmp(j);
                        index = j;
                    }
                }

                confidences[index] = i + 1;
                tmp(index) = 0.0;
            }
        }
/********** DIGIT DRAWER END **********/

/********** SEPERATION START **********/
        DrawLine((28 * pixel_width) + pixel_width, 0, (28 * pixel_width) + pixel_width, ScreenHeight(), {65, 65, 65});
/********** SEPARATION END **********/
        
/********** PREDICTION START **********/
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[0]) + (8 * text_scale * (confidences[0] - 1)), "0: " + to_string(predictions(0)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[1]) + (8 * text_scale * (confidences[1] - 1)), "1: " + to_string(predictions(1)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[2]) + (8 * text_scale * (confidences[2] - 1)), "2: " + to_string(predictions(2)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[3]) + (8 * text_scale * (confidences[3] - 1)), "3: " + to_string(predictions(3)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[4]) + (8 * text_scale * (confidences[4] - 1)), "4: " + to_string(predictions(4)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[5]) + (8 * text_scale * (confidences[5] - 1)), "5: " + to_string(predictions(5)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[6]) + (8 * text_scale * (confidences[6] - 1)), "6: " + to_string(predictions(6)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[7]) + (8 * text_scale * (confidences[7] - 1)), "7: " + to_string(predictions(7)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[8]) + (8 * text_scale * (confidences[8] - 1)), "8: " + to_string(predictions(8)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[9]) + (8 * text_scale * (confidences[9] - 1)), "9: " + to_string(predictions(9)), {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), ScreenHeight() - 25, "by sababot", {75, 75, 75});
/********** PREDICTION END **********/
        
        return true;
	}
};

int main()
{
    load_model(weights1, weights2, bias1, bias2);

	Paint digit_drawer;
	if (digit_drawer.Construct((28 * digit_drawer.pixel_width + digit_drawer.pixel_width) + ((28 * digit_drawer.pixel_width + digit_drawer.pixel_width) / 2), (28 * digit_drawer.pixel_width + digit_drawer.pixel_width), 1, 1))
		digit_drawer.Start();
	return 0;
}
