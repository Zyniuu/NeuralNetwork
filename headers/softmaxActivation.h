#pragma once
#include "neuronActivationPart.h"


class SoftmaxActivation : public Layer
{
private:
	VectorXd outputMatrix;
public:
	VectorXd feedForward(VectorXd inputVals);
	VectorXd backPropagation(VectorXd gradient, double learning_rate);
};
