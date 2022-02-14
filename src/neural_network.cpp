#include "../include/neural_network.h"

// Constructor
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
	this->topology = topology;
	this->learningRate = learningRate;
	
	for (uint i = 0; i < topology.size(); i++)
	{
		if (i == topology.size() - 1)
		{
			neuronLayers.push_back(new RowVector(topology[i]));
		}
		else
		{
			neuronLayers.puch_back(new RowVector(topology[i] + 1));
		}

		cacheLayers.push_back(new RowVector(neuronLayers.size()));
		deltas.push_back(new RowVector(neuronLayers.size()));

		if (i != topology.size() - 1)
		{
			neuronLayers.back()->coeffRef(topology[i]) = 1.0;
			cacheLayers.back()->coeffRef(topology[i]) = 1.0;
		}

		if (i > 0)
		{
			if (i != topology.size() - 1)
			{
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
				weights.back()->setRandom();
				weights.back()->col(topology[i]).setZero();
				weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
			}
			else
			{
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
				weights.back()->setRandom();
			}
		}
	}
}