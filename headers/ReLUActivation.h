#pragma once
#include "neuronActivationPart.h"


class ReLUActivation : public NeuronActivationPart
{
public:
	ReLUActivation() : NeuronActivationPart(&ReLUActivation::reluFunc, &ReLUActivation::reluFuncPrime) {}
	static Matrix<double, Dynamic, 1> reluFunc(Matrix<double, Dynamic, 1> inputVals);
	static Matrix<double, Dynamic, 1> reluFuncPrime(Matrix<double, Dynamic, 1> inputVals);
};
