#include <iostream>
#include "src/neural_network/include/neural_network.h"

using namespace std;

int main()
{
	Eigen::VectorXd inputs(3);
	inputs << 1.1, 5, 4.7;
	Eigen::VectorXd weights(3);
	weights << 0.5, 1.2, 2.1;
	double bias = 2.5;

	cout << calc_output(inputs, weights, bias) << " = output" << endl;
}
