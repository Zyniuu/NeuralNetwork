#pragma once
#include "neuronActivationPart.h"


class SoftmaxActivation : public Layer
{
private:
	VectorXd m_output_matrix;

public:
	VectorXd feedForward(VectorXd input_vals);
	VectorXd backPropagation(VectorXd gradient, double learning_rate);
};
