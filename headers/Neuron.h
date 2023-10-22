#pragma once
#include "NeuronConnection.h"
#include <vector>
#include <cmath>

class Neuron
{
private:
	double outputVal = 1;
	unsigned neuronIndex;
	std::vector<NeuronConnection> outputWeights;
public:
	Neuron(unsigned numOfOutputs, unsigned _neuronIndex);
	void setOutputValue(double value);
	double getOutputValue() const;
	double tanhActivationFunction(double x);
	double activationFunction(double neuronSum);
	void feedForward(std::vector<Neuron>& prevLayer);
};
