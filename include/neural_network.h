#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Eigen>

typedef float Scalar;
typedef Eigen::MatrixXF Matrix;
typedef Eigen::RowVectorXF RowVector;
typedef Eigen::VectorXF ColVector;

class NeuralNetwork
{
public:
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

	void propagateForward(RowVector& input);

	void propagateBackward(RowVector& output);

	void calcErrors(RowVector& output);

	void updateWeights();

	void train(std::vector<RowVector*> data);

	std::vector<RowVector*> neuronLayers;
	std::vector<RowVector*> cacheLayers;
	std::vector<RowVector*> deltas;
	std::vector<Matrix*> weights;
	Scalar learningRate;
};