#include <iostream>
#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/createdata.h"

using namespace std;

int main()
{
	// INPUTS
	Eigen::MatrixXd inputs(3, 4);
	inputs << 	1.0, 2.0, 3.0, 2.5,
				2.0, 5.0, -1.0, 2.0,
				-1.5, 2.7, 3.3, -0.8;

	/* LAYERS
	DenseLayer layer1(4, 5);
	layer1.forward(inputs);
	layer1.outputs = ActivationReLU(layer1.outputs);

	DenseLayer layer2(5, 2);
	layer2.forward(layer1.outputs);
	layer2.outputs = ActivationReLU(layer2.outputs);
	*/

	create_data(100, 3);

	// OUTPUT
	//cout << endl << layer2.outputs << endl << layer2.outputs << endl;
}
