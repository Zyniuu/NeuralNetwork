#pragma once
#include "layer.h"


class NeuronDensePart : public Layer
{
private:
	Matrix<double, Dynamic, Dynamic> m_weightsMatrix;
	Matrix<double, Dynamic, Dynamic> v_weightsMatrix;
	Matrix<double, Dynamic, 1> m_biasMatrix;
	Matrix<double, Dynamic, 1> v_biasMatrix;
	double beta1 = 0.9;
	double beta2 = 0.999;
	double eps = 1e-8;
protected:
	Matrix<double, Dynamic, Dynamic> weightsMatrix;
	Matrix<double, Dynamic, 1> biasMatrix;
public:
	NeuronDensePart(int input_size, int output_size);
	Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals);
	Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate, int epoch);
};
