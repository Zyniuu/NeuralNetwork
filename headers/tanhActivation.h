#pragma once
#include "neuronActivationPart.h"


class TanhActivation : public NeuronActivationPart
{
public:
	TanhActivation() : NeuronActivationPart(&TanhActivation::tanhFunc, &TanhActivation::tanhFuncPrime) {}
	static VectorXd tanhFunc(VectorXd inputVals);
	static VectorXd tanhFuncPrime(VectorXd inputVals);
};
