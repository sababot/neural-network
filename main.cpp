#include <iostream>
#include "src/neural_network/include/neural_network.h"

using namespace std;

int main()
{
	Eigen::VectorXd inputs1(3);
	inputs1 << 1.1, 5, 4.7;
	Eigen::VectorXd weights1(3);
	weights1 << 0.5, 1.2, 2.1;
	double bias1 = 2.5;

	Eigen::VectorXd inputs2(3);
	inputs2 << 1.1, 5, 4.7;
	Eigen::VectorXd weights2(3);
	weights2 << 0.5, 1.2, 2.1;
	double bias2 = 2.5;

	cout << calc_output(inputs1, weights1, bias1) << " = output" << endl;
	cout << calc_output(inputs2, weights2, bias2) << " = output" << endl;
}
