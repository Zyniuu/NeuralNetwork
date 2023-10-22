#include "../headers/Neuron.h"

Neuron::Neuron(unsigned numOfOutputs, unsigned _neuronIndex)
{
	for (int connection = 0; connection < numOfOutputs; connection++)
	{
		outputWeights.push_back(NeuronConnection());
	}
	neuronIndex = _neuronIndex;
}

void Neuron::setOutputValue(double value)
{
	outputVal = value;
}

double Neuron::getOutputValue() const
{
	return outputVal;
}

double Neuron::tanhActivationFunction(double x)
{
	return tanh(x);
}

double Neuron::activationFunction(double neuronSum)
{
	return tanhActivationFunction(neuronSum);
}

void Neuron::feedForward(std::vector<Neuron>& prevLayer)
{
	// sum(i * w)
	double result = 0.0;

	// include bias
	for (int i = 0; i < prevLayer.size(); i++)
	{
		result += prevLayer[i].getOutputValue() * prevLayer[i].outputWeights[neuronIndex].getWeight();
	}

	setOutputValue(activationFunction(result));
}