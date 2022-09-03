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
	DenseLayer layer1(2, 3);
	DenseLayer layer2(3, 3);

	layer1.forward(X);
	layer1.outputs = ActivationReLU(layer1.outputs);

	layer2.forward(layer1.outputs);
	layer2.outputs = ActivationSoftmax(layer2.outputs);

	layer3.forward(layer2.outputs);
	layer3.outputs = ActivationSoftmax(layer3.outputs);

	// OUTPUT
	cout << layer3.outputs << endl;
}
