#pragma once
#include "optimizer.h"


class Layer
{
protected:
	VectorXd m_input_matrix;

public:
	virtual VectorXd feedForward(VectorXd input_vals) = 0;
	virtual VectorXd backPropagation(VectorXd gradient, const Optimizer& optimizer) = 0;
	virtual bool isNeuronDensePart() const { return false; }
};
