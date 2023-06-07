#include "../headers/NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const std::vector<unsigned> topology)
{
	for (int layerNum = 0; layerNum < topology.size(); layerNum++)
	{
		// create new layer and get number of outputs needed
		layers.push_back(std::vector<Neuron>());
		unsigned numOfOutputs;
		// last layer doesn't have any outputs
		if (layerNum == topology.size() - 1)
			numOfOutputs = 0;
		else
			numOfOutputs = topology[layerNum + 1];
		// fill up new layer with neurons
		for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			layers.back().push_back(Neuron(numOfOutputs));
		}
	}
}
