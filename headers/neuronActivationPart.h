#pragma once
#include <cmath>
#include "layer.h"


class NeuronActivationPart : public Layer
{
protected:
	VectorXd(*activationFunc)(VectorXd);
	VectorXd(*activationFuncPrime)(VectorXd);

	NeuronActivationPart(VectorXd (*_activationFunc)(VectorXd), VectorXd (*_activationFuncPrime)(VectorXd));
	VectorXd feedForward(VectorXd inputVals);
	VectorXd backPropagation(VectorXd gradient, double learning_rate);
};
