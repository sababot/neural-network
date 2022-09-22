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

	// FORWARD
	dense_layer layer1(2, 3);
	activation_relu activation1;

	dense_layer layer2(3, 3);
	activation_softmax_loss_categoral_cross_entropy loss_activation;

	layer1.forward(X);
	activation1.forward(layer1.outputs);

	layer2.forward(activation1.outputs);
	double loss = loss_activation.forward(layer2.outputs, y);

	// BACKWARD
	loss_activation.backward(loss_activation.outputs, y);
	layer2.backward(loss_activation.dinputs);
	activation1.backward(layer2.dinputs);
	layer1.backward(activation1.dinputs);

	optimizer_sgd optimization(0.9);

	cout << layer1.weights << endl << endl;
	cout << layer1.biases << endl << endl;
	cout << layer2.weights << endl << endl;
	cout << layer2.biases << endl << endl;

	optimization.update_params(&layer1);
	optimization.update_params(&layer2);

	cout << layer1.weights << endl << endl;
	cout << layer1.biases << endl << endl;
	cout << layer2.weights << endl << endl;
	cout << layer2.biases << endl << endl;

	return 0;
}
