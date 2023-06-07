#pragma once
#include "Neuron.h"
#include <vector>

class NeuralNetwork
{
private:
	std::vector<std::vector<Neuron>> layers;
public:
	NeuralNetwork(const std::vector<unsigned> topology);
};
