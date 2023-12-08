#include "../headers/neuronActivationPart.h"


NeuronActivationPart::NeuronActivationPart(VectorXd (*_activationFunc)(VectorXd), VectorXd (*_activationFuncPrime)(VectorXd))
{
	activationFunc = _activationFunc;
	activationFuncPrime = _activationFuncPrime;
}

VectorXd NeuronActivationPart::feedForward(VectorXd inputVals)
{
	inputMatrix = inputVals;
	return activationFunc(inputMatrix);
}

VectorXd NeuronActivationPart::backPropagation(VectorXd gradient, double learning_rate)
{
	return gradient.cwiseProduct(activationFuncPrime(inputMatrix));
}