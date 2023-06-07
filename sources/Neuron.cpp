#include "../headers/Neuron.h"

Neuron::Neuron(unsigned numOfOutputs)
{
	for (int connection = 0; connection < numOfOutputs; connection++)
	{
		outputWeights.push_back(NeuronConnection());
	}
}
