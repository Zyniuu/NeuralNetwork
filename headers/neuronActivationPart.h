#pragma once
#include <cmath>
#include "layer.h"


class NeuronActivationPart : public Layer
{
protected:
	VectorXd(*m_activation_func)(VectorXd);
	VectorXd(*m_activation_func_prime)(VectorXd);

	NeuronActivationPart(VectorXd (*activation_func)(VectorXd), VectorXd (*activation_func_prime)(VectorXd));
	VectorXd feedForward(VectorXd input_vals);
	VectorXd backPropagation(VectorXd gradient, const Optimizer& optimizer);
};
