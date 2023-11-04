#include "../headers/neuronActivationPart.h"


NeuronActivationPart::NeuronActivationPart(Matrix<double, Dynamic, 1> (*_activationFunc)(Matrix<double, Dynamic, 1>), Matrix<double, Dynamic, 1>(*_activationFuncPrime)(Matrix<double, Dynamic, 1>))
{
	activationFunc = _activationFunc;
	activationFuncPrime = _activationFuncPrime;
}

Matrix<double, Dynamic, 1> NeuronActivationPart::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	inputMatrix = inputVals;
	return activationFunc(inputMatrix);
}

Matrix<double, Dynamic, 1> NeuronActivationPart::backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate, int epoch)
{
	return gradient.cwiseProduct(activationFuncPrime(inputMatrix));
}