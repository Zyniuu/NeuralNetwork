#pragma once
#include "neuronActivationPart.h"


class SoftmaxActivation : public Layer
{
private:
	VectorXd outputMatrix;
public:
	Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate);
};
