#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

#include "../src/neural_network/include/layers.h"
#include "../src/neural_network/include/activation.h"
#include "../src/utils/include/import_data.h"

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Eigen>

using namespace std;

// init model
layer_single layer1(784, 64);
activation_relu activation1;

layer_single layer2(64, 10);
activation_softmax activation2;

// parameters
Eigen::MatrixXd weights1 = read_data("../model/weights1.csv", 784, 64);
Eigen::MatrixXd weights2 = read_data("../model/weights2.csv", 64, 10);
Eigen::VectorXd bias1 = read_data_single("../model/biases1.csv", 64);
Eigen::VectorXd bias2 = read_data_single("../model/biases2.csv", 10);

void load_model(Eigen::MatrixXd w1, Eigen::MatrixXd w2, Eigen::VectorXd b1, Eigen::VectorXd b2)
{
    layer1.weights_single = w1;
    layer2.weights_single = w2;

    layer1.biases_single = b1;
    layer2.biases_single = b2;
}

double output(Eigen::VectorXd inputs)
{
    layer1.forward_single(inputs);
    activation1.forward_single(layer1.outputs_single);
    layer2.forward_single(activation1.outputs_single);
    activation2.forward_single(layer2.outputs_single);

    double maximum = activation2.outputs_single.maxCoeff();
    cout << maximum << endl;
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
    std::vector<std::array<int, 2>> pixels;
    int x = 0;
    int y = 0;
    int text_scale = 3;
    int confidences[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int colours[10][3] = {{255, 255, 255},
                          {235, 235, 235},
                          {215, 215, 215},
                          {195, 195, 195},
                          {175, 175, 175},
                          {155, 155, 155},
                          {135, 135, 135},
                          {115, 115, 115},
                          {95, 95, 95},
                          {75, 75, 75}};

public:
	bool OnUserCreate() override
	{
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
        // Erase previous frame
	    Clear(olc::BLACK);

/********** DIGIT DRAWER START **********/
        // Draw
        Eigen::VectorXd inputs(784);

        for (int i = 0; i < inputs.rows(); i++)
        {
            //
        }

        for (int i = 0; i < pixels.size(); i++)
        {
            if ((int(pixels[i][0]) % pixel_width == 0) && (int(pixels[i][1]) % pixel_width == 0))
                FillRect(int(pixels[i][0]), int(pixels[i][1]), pixel_width, pixel_width, olc::WHITE);
        }

        // Cursor
        FillRect(int(x), int(y), pixel_width, pixel_width, olc::WHITE);
        
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

            //output(inputs);
        }
        
        // Erase
        if (GetMouse(1).bHeld && GetMouseY() <= (28 * pixel_width) + (pixel_width - 1) && GetMouseX() <= (28 * pixel_width) + (pixel_width - 1))
        {
            for (int i = 0; i < pixels.size(); i++)
            {
                if (pixels[i][0] == x && pixels[i][1] == y)
                    pixels.erase(pixels.begin() + i);
            }
        }
/********** DIGIT DRAWER END **********/

/********** SEPERATION START **********/
        DrawLine((28 * pixel_width) + pixel_width, 0, (28 * pixel_width) + pixel_width, ScreenHeight(), olc::WHITE);
/********** SEPARATION END **********/
        
/********** PREDICTION START **********/
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[0]) + (8 * text_scale * (confidences[0] - 1)), "0: 83%", {255, 255, 255}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[1]) + (8 * text_scale * (confidences[1] - 1)), "1: 15%", {235, 235, 235}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[2]) + (8 * text_scale * (confidences[2] - 1)), "2: 01%", {215, 215, 215}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[3]) + (8 * text_scale * (confidences[3] - 1)), "3: 01%", {195, 195, 195}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[4]) + (8 * text_scale * (confidences[4] - 1)), "4: 00%", {175, 175, 175}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[5]) + (8 * text_scale * (confidences[5] - 1)), "5: 00%", {155, 155, 155}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[6]) + (8 * text_scale * (confidences[6] - 1)), "6: 00%", {135, 135, 135}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[7]) + (8 * text_scale * (confidences[7] - 1)), "7: 00%", {115, 115, 115}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[8]) + (8 * text_scale * (confidences[8] - 1)), "8: 00%", {95, 95, 95}, text_scale);
        DrawString(ScreenWidth() - ((ScreenWidth() - ((28 * pixel_width) + pixel_width)) - 20), (10 * confidences[9]) + (8 * text_scale * (confidences[9] - 1)), "9: 00%", {75, 75, 75}, text_scale);
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
