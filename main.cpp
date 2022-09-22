#include <iostream>
#include <algorithm>

#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/neural_network/include/loss.h"
#include "src/neural_network/include/optimization.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/import_data.h"

using namespace std;

int main()
{
	// DATASET
	Eigen::MatrixXd X = read_data("datasets/vertical/data.csv", 300, 2);
	Eigen::VectorXd y = read_data_single("datasets/vertical/targets.csv", 300);
	Eigen::MatrixXd y_onehot = convert_to_onehot(y, 3);

	// VARIABLES
	int epochs = 10000;
	double learning_rate = 0.99;

	// MODEL DEFINITION
	dense_layer layer1(2, 64);
	activation_relu activation1;

	dense_layer layer2(64, 3);
	activation_softmax_loss_categoral_cross_entropy loss_activation;

	optimizer_sgd optimizer(learning_rate);

	for (int i = 0; i <= epochs; i++)
	{
		// FORWARD
		layer1.forward(X);
		activation1.forward(layer1.outputs);

		layer2.forward(activation1.outputs);
		double loss = loss_activation.forward(layer2.outputs, y);

		// OUTPUT INFORMATION
		if (i % 10 == 0)
		{
			cout << "==================================================>" <<
			endl << "epoch: " << i << "/" << epochs <<
			endl << "accuracy: " << calculate_accuracy(loss_activation.outputs, y)
		    << "	loss: " << loss << endl;
		}

		// BACKWARD
		loss_activation.backward(loss_activation.outputs, y);
		layer2.backward(loss_activation.dinputs);

		activation1.backward(layer2.dinputs);
		layer1.backward(activation1.dinputs);

		// OPTIMIZATION
		optimizer.update_params(&layer1);
		optimizer.update_params(&layer2);
	}

	// FINAL RESULT OUTPUT
	cout << "==================================================>" << endl;
	cout << endl << "final result:" 
	<< endl << "accuracy: " << calculate_accuracy(loss_activation.outputs, y) 
	<< "	loss: " << loss_activation.forward(layer2.outputs, y) << endl;

	return 0;
}
