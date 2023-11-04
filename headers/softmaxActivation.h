#pragma once
#include "neuronActivationPart.h"


class SoftmaxActivation : public NeuronActivationPart
{
public:
	SoftmaxActivation() : NeuronActivationPart(&SoftmaxActivation::softmaxFunc, &SoftmaxActivation::softmaxPrime) {}
	static Matrix<double, Dynamic, 1> softmaxFunc(Matrix<double, Dynamic, 1> inputVals);
	static Matrix<double, Dynamic, 1> softmaxPrime(Matrix<double, Dynamic, 1> inputVals);
};
