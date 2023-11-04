#pragma once
#include "neuronActivationPart.h"


class SigmoidActivation : public NeuronActivationPart
{
public:
	SigmoidActivation() : NeuronActivationPart(&SigmoidActivation::sigmoidFunc, &SigmoidActivation::sigmoidFuncPrime) {}
	static Matrix<double, Dynamic, 1> sigmoidFunc(Matrix<double, Dynamic, 1> inputVals);
	static Matrix<double, Dynamic, 1> sigmoidFuncPrime(Matrix<double, Dynamic, 1> inputVals);
};
