#include "../headers/neuralNetwork.h"


NeuralNetwork::NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction)
{
	_topology = topology;
	if (!_topology.empty())
		labels = _topology.back();
	switch (activationFunction)
	{
	case TANH:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<TanhActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<TanhActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<TanhActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<TanhActivation, SoftmaxActivation>();
			break;
		}
		break;
	case RELU:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<ReLUActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<ReLUActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<ReLUActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<ReLUActivation, SoftmaxActivation>();
			break;
		}
		break;
	case SIGMOID:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<SigmoidActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<SigmoidActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<SigmoidActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<SigmoidActivation, SoftmaxActivation>();
			break;
		}
		break;
	case SOFTMAX:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<SoftmaxActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<SoftmaxActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<SoftmaxActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<SoftmaxActivation, SoftmaxActivation>();
			break;
		}
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

Matrix<double, Dynamic, 1> NeuralNetwork::labelToEigenMatrix(int label)
{
	Matrix<double, Dynamic, 1> result = MatrixXd::Zero(labels, 1);
	if (label >= 0 && label < labels)
		result(label, 0) = 1.0;
	return result;
}

template <class T, class Z>
void NeuralNetwork::fillLayers()
{
	std::vector<unsigned>::iterator iter = _topology.begin();
	for (iter; iter < _topology.end() - 1; iter++)
	{
		layers.push_back(new NeuronDensePart(*iter, *(iter + 1)));
		if (std::next(iter) == _topology.end() - 1)
			layers.push_back(new Z());
		else
			layers.push_back(new T());
	}
}

Matrix<double, Dynamic, 1> NeuralNetwork::predict(Matrix<double, Dynamic, 1> inputVals)
{
	Matrix<double, Dynamic, 1> output = normalizeVector(inputVals);
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

Matrix<double, Dynamic, 1> NeuralNetwork::predict(std::vector<double> inputVals)
{
	Matrix<double, Dynamic, 1> output = vectorToEigenMatrix(inputVals);
	output = normalizeVector(output);
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

Matrix<double, Dynamic, 1> NeuralNetwork::normalizeVector(Matrix<double, Dynamic, 1>& vectorToNormalize)
{
	double minVal = vectorToNormalize.minCoeff();
	double maxVal = vectorToNormalize.maxCoeff();
	return (vectorToNormalize.array() - minVal) / (maxVal - minVal);
}

double NeuralNetwork::meanSquaredError(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output)
{
	Matrix<double, Dynamic, 1> _error = (true_output - predicted_output).array().square();
	return _error.mean();
}

Matrix<double, Dynamic, 1> NeuralNetwork::meanSquaredErrorPrime(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output)
{
	double scale = 2.0 / true_output.size();
	return scale * (predicted_output - true_output);
}

void NeuralNetwork::train(CSVParser& parser, int epochs, double learning_rate)
{
	int numberOfSamples = parser.countLines();
	for (int i = 0; i < epochs; i++)
	{
		double _error = 0.0;
		while (!parser.endOfFile())
		{
			parser.getDataFromSingleLine();
			Matrix<double, Dynamic, 1> inputVals = vectorToEigenMatrix(parser.getValues());
			Matrix<double, Dynamic, 1> output = predict(inputVals);
			Matrix<double, Dynamic, 1> y;
			if (_topology.back() > 1)
				y = labelToEigenMatrix(parser.getTarget());
			else
			{
				y.resize(1);
				y(0) = parser.getTarget();
			}
			_error += meanSquaredError(y, output);
			Matrix<double, Dynamic, 1> gradient = meanSquaredErrorPrime(y, output);
			for (auto iter = layers.rbegin(); iter != layers.rend(); ++iter)
			{
				gradient = (*iter)->backPropagation(gradient, learning_rate, i);
			}
		}
		_error /= numberOfSamples;
		std::cout << i + 1 << "/" << epochs << "\terror = " << _error << std::endl;
		parser.restartFile();
	}
}
