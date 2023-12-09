#pragma once
#include "neuronActivationPart.h"


class SigmoidActivation : public NeuronActivationPart
{
public:
	SigmoidActivation() : NeuronActivationPart(&SigmoidActivation::sigmoidFunc, &SigmoidActivation::sigmoidFuncPrime) {}
	static VectorXd sigmoidFunc(VectorXd input_vals);
	static VectorXd sigmoidFuncPrime(VectorXd input_vals);
};
