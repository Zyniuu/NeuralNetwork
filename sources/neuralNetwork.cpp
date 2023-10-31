#include "../headers/neuralNetwork.h"


NeuralNetwork::NeuralNetwork(std::vector<unsigned>& topology, int activationFunction)
{
	_topology = topology;
	switch (activationFunction)
	{
	case TANH:
		fillLayers<TanhActivation>();
		break;
	}
}

NeuralNetwork::~NeuralNetwork()
{
	for (Layer* layer : layers) 
	{
		delete layer;
	}
	layers.clear();
}

template <class T>
void NeuralNetwork::fillLayers()
{
	std::vector<unsigned>::iterator iter = _topology.begin();
	for (iter; iter < _topology.end() - 1; iter++)
	{
		layers.push_back(new NeuronDensePart(*iter, *(iter + 1)));
		layers.push_back(new T());
	}
}

Matrix<double, Dynamic, 1> NeuralNetwork::predict(Matrix<double, Dynamic, 1> inputVals)
{
	Matrix<double, Dynamic, 1> output = inputVals;
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

Matrix<double, Dynamic, 1> NeuralNetwork::vectorToEigenMatrix(const std::vector<double>& inputVector)
{
	Map<const VectorXd> eigenMap(inputVector.data(), inputVector.size());
	VectorXd eigenMatrix = eigenMap;
	return eigenMatrix;
}

void NeuralNetwork::train(CSVParser& parser)
{
	while (!parser.endOfFile())
	{
		parser.getDataFromSingleLine();
		Matrix<double, Dynamic, 1> inputVals = vectorToEigenMatrix(parser.getValues());
		std::cout << inputVals << "\n\n";
		predict(inputVals);
		std::cout << "\n\n\n\n";
	}
}
