#include <iostream>
#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/neural_network/include/loss.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/import_data.h"

using namespace std;

int main()
{
	// DATASET
	Eigen::MatrixXd X = read_data("datasets/vertical/data.csv", 300, 2);
	Eigen::VectorXd y = read_data_single("datasets/vertical/targets.csv", 300);
	Eigen::MatrixXd y_onehot = convert_to_onehot(y, 3);

	// LAYERS
	dense_layer layer1(2, 3);
	dense_layer layer2(3, 3);

	// OPTIMIZATION
	double lowest_loss = 999;
	Eigen::MatrixXd best_layer1_weights = layer1.weights;
	Eigen::VectorXd best_layer1_biases = layer1.biases;
	Eigen::MatrixXd best_layer2_weights = layer2.weights;
	Eigen::VectorXd best_layer2_biases = layer2.biases;

	// EPOCHS
	for (int i = 0; i < 10000; i++)
	{
		// randomly adjusting best weights and biases
		layer1.weights += Eigen::MatrixXd::Random(layer1.n_inputs, layer1.n_neurons) * 0.05;
		layer1.biases += Eigen::VectorXd::Random(layer1.n_neurons) * 0.05;
		layer2.weights += Eigen::MatrixXd::Random(layer2.n_inputs, layer2.n_neurons) * 0.05;
		layer2.biases += Eigen::VectorXd::Random(layer2.n_neurons) * 0.05;

		// launch nural network
		layer1.forward(X);
		layer1.outputs = activation_relu(layer1.outputs);
		layer2.forward(layer1.outputs);
		layer2.outputs = activation_softmax(layer2.outputs);

		// calculate loss
		double loss = calculate_loss(layer2.outputs, y);

		// check if loss is better than best loss
		if (loss < lowest_loss)
		{
			cout << "[NEW] epoch: " << i << ", loss: " << loss << endl;

			best_layer1_weights = layer1.weights;
			best_layer1_biases = layer1.biases;
			best_layer2_weights = layer2.weights;
			best_layer2_biases = layer2.biases;

			lowest_loss = loss;
		}

		else
		{
			layer1.weights = best_layer1_weights;
			layer1.biases = best_layer1_biases;
			layer2.weights = best_layer2_weights;
			layer2.biases = best_layer2_biases;
		}
	}

	return 0;
}
