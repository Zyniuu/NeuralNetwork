#include "../headers/neuronActivationPart.h"


NeuronActivationPart::NeuronActivationPart(Matrix<double, Dynamic, 1> (*_activationFunc)(Matrix<double, Dynamic, 1>))
{
	activationFunc = _activationFunc;
}

Matrix<double, Dynamic, 1> NeuronActivationPart::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	inputMatrix = inputVals;
	return activationFunc(inputMatrix);
}
