#pragma once
#include <vector>
#include <iostream>


class Neuron{};

class NeuralNetwork
{
private:
	std::vector< std::vector<Neuron> > layers;
public:
	NeuralNetwork(const std::vector<unsigned> topology);
};