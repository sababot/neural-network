#include <iostream>
#include <algorithm>

#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/neural_network/include/loss.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/import_data.h"

using namespace std;

int main()
{
	/*
	Eigen::MatrixXd softmax_outputs(3, 3);
	softmax_outputs << 0.7, 0.1, 0.2,
					   0.1, 0.5, 0.4,
					   0.02, 0.9, 0.08;

	Eigen::VectorXd class_targets(3);
	class_targets << 0, 1, 1;

	activation_softmax_loss_categoral_cross_entropy softmax_loss;
	softmax_loss.backward(softmax_outputs, class_targets);
	Eigen::MatrixXd dvalues1 = softmax_loss.dinputs;

	activation_softmax activation;
	activation.outputs = softmax_outputs;
	loss_categoral_cross_entropy loss;
	loss.backward(softmax_outputs, class_targets);
	activation.backward(loss.dinputs);
	Eigen::MatrixXd dvalues2 = activation.dinputs;

	cout << dvalues2 << endl;
	*/

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

	cout << "loss: " << loss << endl << endl;

	// BACKWARD
	loss_activation.backward(loss_activation.outputs, y);
	layer2.backward(loss_activation.dinputs);
	activation1.backward(layer2.dinputs);
	layer1.backward(activation1.dinputs);

	cout << layer2.dweights << endl;

	/*
	cout << layer1.dweights << endl << endl;
	cout << layer1.dbiases << endl << endl;
	cout << layer2.dweights << endl << endl;
	cout << layer2.dbiases << endl << endl;
	*/

	return 0;
}
