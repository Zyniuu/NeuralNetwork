#pragma once
#include "layer.h"


class NeuronDensePart : public Layer
{
protected:
	Matrix<double, Dynamic, Dynamic> weightsMatrix;
	Matrix<double, Dynamic, 1> biasMatrix;
public:
	NeuronDensePart(int input_size, int output_size);
	Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate);
};
