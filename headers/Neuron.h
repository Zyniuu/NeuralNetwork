#pragma once
#include "NeuronConnection.h"
#include <vector>

class Neuron
{
private:
	double outputVal;
	std::vector<NeuronConnection> outputWeights;
public:
	Neuron(unsigned numOfOutputs);
};
