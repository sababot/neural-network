#include <iostream>
#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/import_data.h"

using namespace std;

int main()
{
	// INPUTS
	Eigen::MatrixXd X = read_data("datasets/spiral/data.csv", 300, 2);

	/* LAYERS */
	DenseLayer layer1(2, 5);
	DenseLayer layer2(5, 4);

	layer1.forward(X);
	layer1.outputs = ActivationReLU(layer1.outputs);

	layer2.forward(layer1.outputs);
	layer2.outputs = ActivationReLU(layer2.outputs);

	layer3.forward(layer2.outputs);
	layer3.outputs = ActivationReLU(layer3.outputs);

	// OUTPUT
	cout << layer1.outputs << endl << endl << layer2.outputs << endl << layer3.outputs << endl;
}
