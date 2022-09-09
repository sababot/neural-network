#include <iostream>
#include "src/neural_network/include/layers.h"
#include "src/neural_network/include/activation.h"
#include "src/neural_network/include/loss.h"
#include "src/utils/include/utils.h"
#include "src/utils/include/import_data.h"

using namespace std;

int main()
{
	/* INPUTS */
	Eigen::MatrixXd X = read_data("datasets/spiral/data.csv", 300, 2);
	Eigen::VectorXd y = read_data_single("datasets/spiral/targets.csv", 300);
	Eigen::MatrixXd y_onehot = convert_to_onehot(y, 3);

	/* TEST MATRIX */
	Eigen::MatrixXd X_test(3, 3);
	X_test << 0.7, 0.1, 0.2,
	          0.1, 0.5, 0.4,
	          0.02, 0.9, 0.08;

	/* LAYERS */
	dense_layer layer1(2, 3);
	dense_layer layer2(3, 3);

	layer1.forward(X);
	layer1.outputs = activation_relu(layer1.outputs);

	layer2.forward(layer1.outputs);
	layer2.outputs = activation_softmax(layer2.outputs);


	/* OUTPUT */
	cout << calculate_loss(layer2.outputs, y) << endl;

	return 0;
}
