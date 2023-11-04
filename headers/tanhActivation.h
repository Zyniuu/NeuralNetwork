#pragma once
#include "neuronActivationPart.h"


class TanhActivation : public NeuronActivationPart
{
public:
	TanhActivation() : NeuronActivationPart(&TanhActivation::tanhFunc, &TanhActivation::tanhFuncPrime) {}
	static Matrix<double, Dynamic, 1> tanhFunc(Matrix<double, Dynamic, 1> inputVals);
	static Matrix<double, Dynamic, 1> tanhFuncPrime(Matrix<double, Dynamic, 1> inputVals);
};
