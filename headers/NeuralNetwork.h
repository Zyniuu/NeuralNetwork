#pragma once
#include "Neuron.h"
#include <vector>
#include <iostream>

class NeuralNetwork
{
private:
	std::vector<std::vector<Neuron>> layers;
public:
	NeuralNetwork(const std::vector<unsigned>& topology);
	void feedForfward(const std::vector<double>& inputValues);
	void backPropagation(const std::vector<double>& targetValues);
	void getResults(std::vector<double>& resultVals) const;
	void printNeuronOutputs();
};
