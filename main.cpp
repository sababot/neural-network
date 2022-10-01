#include <iostream>
#include <algorithm>
#include <eigen3/Eigen/Eigen>

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
	Eigen::MatrixXd X = read_data("datasets/digit-recognizer/processed/data_test.csv", 40000, 784);
	Eigen::VectorXd y = read_data_single("datasets/digit-recognizer/processed/targets.csv", 40000);
	Eigen::MatrixXd X_test = read_data("datasets/digit-recognizer/processed/test_data.csv", 1000, 784);
	Eigen::MatrixXd y_test = read_data_single("datasets/digit-recognizer/processed/test_targets.csv", 1000);
	
	Eigen::MatrixXd y_onehot = convert_to_onehot(y, 10);
	cout << "[DATA IMPORTED]" << endl;

	// VARIABLES
	int epochs = 50;
	double learning_rate = 0.001;
	double decay = 0;
	//double momentum = 0.9;
	double epsilon = 0.0000001;
	//double rho = 0.999;
	double beta_1 = 0.9;
	double beta_2 = 0.999;

	// MODEL DEFINITION
	dense_layer layer1(784, 64);
	activation_relu activation1;

	dense_layer layer2(64, 10);
	activation_softmax_loss_categoral_cross_entropy loss_activation;

	//optimizer_sgd optimizer(learning_rate, decay, momentum);
	//optimizer_adagrad optimizer(learning_rate, decay, epsilon);
	//optimizer_rmsprop optimizer(learning_rate, decay, epsilon, rho);
	optimizer_adam optimizer(learning_rate, decay, epsilon, beta_1, beta_2);

	cout << "[MODEL DEFINED]" << endl;

	cout << "[TRAINING STARTED]" << endl;
	for (int i = 0; i <= epochs; i++)
	{
		// FORWARD
		layer1.forward(X);
		activation1.forward(layer1.outputs);

		layer2.forward(activation1.outputs);
		double loss = loss_activation.forward(layer2.outputs, y);

		// OUTPUT INFORMATION
		if (i % 1 == 0)
		{
			cout << "==================================================>" <<
			endl << "epoch: " << i << "/" << epochs << 
			endl << "--> lr      : " << optimizer.current_learning_rate <<
			endl << "--> accuracy: " << calculate_accuracy(loss_activation.outputs, y) <<
		    endl << "--> loss    : " << loss <<
		    endl << "==================================================>" << endl << endl;
		}

		// BACKWARD
		loss_activation.backward(loss_activation.outputs, y);
		layer2.backward(loss_activation.dinputs);

		activation1.backward(layer2.dinputs);
		layer1.backward(activation1.dinputs);

		// OPTIMIZATION
		optimizer.pre_update_params();
		optimizer.update_params(&layer1);
		optimizer.update_params(&layer2);
		optimizer.post_update_params();
	}

	// FINAL RESULT OUTPUT
	layer1.forward(X_test);
	activation1.forward(layer1.outputs);

	layer2.forward(activation1.outputs);
	double loss = loss_activation.forward(layer2.outputs, y_test);
	
	cout << "final result:" 
	<< endl << "accuracy: " << calculate_accuracy(loss_activation.outputs, y) 
 	<< "	loss: " << loss_activation.forward(layer2.outputs, y_test) << endl;

 	export_data(layer1.weights.transpose(), "model/weights1.csv");
 	export_data(layer1.biases.transpose(), "model/biases1.csv");
 	export_data(layer2.weights, "model/weights2.csv");
 	export_data(layer2.biases, "model/biases2.csv");

	return 0;
}
