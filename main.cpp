#include <iostream>
#include "src/neural_network/include/neural_network.h"

using namespace std;

int main()
{
	// INPUTS
	Eigen::MatrixXd inputs(3, 4);
	inputs << 	1.0, 2.0, 3.0, 2.5,
				2.0, 5.0, -1.0, 2.0,
				-1.5, 2.7, 3.3, -0.8;

	/* WEIGHTS
	Eigen::MatrixXd weights(3, 4);
	weights <<   0.2,   0.8,  -0.5,  1.0,
				 0.5,  -0.91, 0.26, -0.5,
				-0.26, -0.27, 0.17, 0.87;

	// BIASES
	Eigen::VectorXd biases(3);
	biases << 	2, 3, 0.5;

	// OUTPUTS
	Eigen::MatrixXd outputs(3, 3);
	outputs = (inputs * weights.transpose()) + biases.replicate(1, 3).transpose(); // biases converted to matrix to fit shape of output

	// PRINT
	cout << outputs << endl;
	*/

	DenseLayer layer1(4, 5);
	DenseLayer layer2(5, 2);

	layer1.forward(inputs);
	cout << endl << layer1.outputs << endl;

	layer2.forward(layer1.outputs);
	cout << endl << layer2.outputs << endl;
}
