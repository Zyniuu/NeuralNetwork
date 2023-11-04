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

	Matrix<double, Dynamic, 1> vectorToEigenMatrix(const std::vector<double>& inputVector);
	Matrix<double, Dynamic, 1> labelToEigenMatrix(int label);
	Matrix<double, Dynamic, 1> normalizeVector(Matrix<double, Dynamic, 1>& vectorToNormalize);
	template <class T, class Z>
	void fillLayers();
public:
	NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction);
	~NeuralNetwork();
	Matrix<double, Dynamic, 1> predict(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> predict(std::vector<double> inputVals);
	double meanSquaredError(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output);
	Matrix<double, Dynamic, 1> meanSquaredErrorPrime(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output);
	void train(CSVParser& parser, int epochs, double learning_rate);
};
