#include "../headers/NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(const std::vector<unsigned>& topology)
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
			layers.back().push_back(Neuron(numOfOutputs, neuronNum));
		}
	}
}

void NeuralNetwork::feedForfward(const std::vector<double>& inputValues)
{
	// check if provided number of values is the same as number of inputs in NeuralNet
	if (inputValues.size() != layers[0].size() - 1)
	{
		std::cout << "Incorrect number of input values" << std::endl;
		exit(1);
	}

	// Assign input values -> input neurons
	for (int i = 0; i < inputValues.size(); i++)
		layers[0][i].setOutputValue(inputValues[i]);

	for (int layerNum = 1; layerNum < layers.size(); layerNum++)
	{
		std::vector<Neuron>& prevLayer = layers[layerNum - 1];
		for (int i = 0; i < layers[layerNum].size() - 1; i++)
		{
			layers[layerNum][i].feedForward(prevLayer);
		}
	}
}

void NeuralNetwork::printNeuronOutputs()
{
	for (int i = 0; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i].size(); j++)
		{
			std::cout << "Layer: " << i << std::endl;
			std::cout << "Neuron: " << j << std::endl;
			std::cout << "Value: " << layers[i][j].getOutputValue() << "\n\n";
		}
	}
}