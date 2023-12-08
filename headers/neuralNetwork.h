#pragma once
#include "neuronDensePart.h"
#include "tanhActivation.h"
#include "ReLUActivation.h"
#include "sigmoidActivation.h"
#include "softmaxActivation.h"
#include "CSVParser.h"


const int TANH = 0;
const int RELU = 1;
const int SIGMOID = 2;
const int SOFTMAX = 3;


class NeuralNetwork
{
private:
	std::vector<Layer*> layers;
	std::vector<unsigned> _topology;
	int labels;
	int _activationFunction;
	int _outputActivationFunction;
	double maxInputValue;
	double minInputValue;

	VectorXd vectorToEigenMatrix(const std::vector<double>& inputVector);
	VectorXd labelToEigenMatrix(int label);
	VectorXd normalizeVector(VectorXd& vectorToNormalize);
	template <class T, class Z>
	void fillLayers();
public:
	NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction);
	~NeuralNetwork();
	VectorXd predict(VectorXd inputVals);
	VectorXd predict(std::vector<double> inputVals);
	double meanSquaredError(VectorXd true_output, VectorXd predicted_output);
	VectorXd meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output);
	void train(CSVParser& parser, int epochs, double learning_rate);
	bool saveNetworkToFile(std::string filename);
};
