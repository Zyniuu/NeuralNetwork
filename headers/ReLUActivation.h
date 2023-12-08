#pragma once
#include "neuronActivationPart.h"


class ReLUActivation : public NeuronActivationPart
{
public:
	ReLUActivation() : NeuronActivationPart(&ReLUActivation::reluFunc, &ReLUActivation::reluFuncPrime) {}
	static VectorXd reluFunc(VectorXd inputVals);
	static VectorXd reluFuncPrime(VectorXd inputVals);
};
