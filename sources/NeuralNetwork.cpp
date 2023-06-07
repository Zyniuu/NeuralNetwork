#include "../headers/NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const std::vector<unsigned> topology)
{
	for (int layerNum = 0; layerNum < topology.size(); layerNum++)
	{
		// create new layer
		layers.push_back(std::vector<Neuron>());
		// fill up new layer with neurons
		for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			layers.back().push_back(Neuron());
			std::cout << "Created a neuron!" << std::endl;
		}
	}
}