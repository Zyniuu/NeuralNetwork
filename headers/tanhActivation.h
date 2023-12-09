#pragma once
#include "neuronActivationPart.h"


class TanhActivation : public NeuronActivationPart
{
public:
	TanhActivation() : NeuronActivationPart(&TanhActivation::tanhFunc, &TanhActivation::tanhFuncPrime) {}
	static VectorXd tanhFunc(VectorXd input_vals);
	static VectorXd tanhFuncPrime(VectorXd input_vals);
};
