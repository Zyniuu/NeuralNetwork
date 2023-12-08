#pragma once
#include "neuronActivationPart.h"


class SigmoidActivation : public NeuronActivationPart
{
public:
	SigmoidActivation() : NeuronActivationPart(&SigmoidActivation::sigmoidFunc, &SigmoidActivation::sigmoidFuncPrime) {}
	static VectorXd sigmoidFunc(VectorXd inputVals);
	static VectorXd sigmoidFuncPrime(VectorXd inputVals);
};
