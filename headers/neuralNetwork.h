#pragma once
#include "neuronDensePart.h"
#include "tanhActivation.h"
#include "CSVParser.h"


const int TANH = 0;


class NeuralNetwork
{
private:
	std::vector<Layer*> layers;
	std::vector<unsigned> _topology;

	Matrix<double, Dynamic, 1> predict(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> vectorToEigenMatrix(const std::vector<double>& inputVector);
	template <class T>
	void fillLayers();
public:
	NeuralNetwork(std::vector<unsigned>& topology, int activationFunction);
	~NeuralNetwork();
	void train(CSVParser<double>& parser);
};
