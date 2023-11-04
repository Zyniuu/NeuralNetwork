#pragma once
#include <cmath>
#include "layer.h"


class NeuronActivationPart : public Layer
{
protected:
	Matrix<double, Dynamic, 1> (*activationFunc)(Matrix<double, Dynamic, 1>);
	Matrix<double, Dynamic, 1> (*activationFuncPrime)(Matrix<double, Dynamic, 1>);

	NeuronActivationPart(Matrix<double, Dynamic, 1> (*_activationFunc)(Matrix<double, Dynamic, 1>), Matrix<double, Dynamic, 1> (*_activationFuncPrime)(Matrix<double, Dynamic, 1>));
	Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate, int epoch);
};
