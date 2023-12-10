#pragma once
#include "optimizer.h"


class SGD : public Optimizer
{
private:
	double m_momentum;
	Matrix<double, Dynamic, Dynamic> m_momentum_weights;
	VectorXd m_momentum_bias;

public:
	SGD(double learning_rate = 0.01, double momentum = 0.9);
	SGD(const SGD& other, int weights_matrix_rows, int weights_matrix_cols);
	Matrix<double, Dynamic, Dynamic> getDeltaWeights(VectorXd input_vector, VectorXd gradient);
	VectorXd getDeltaBias(VectorXd gradient);
	SGD* clone(int weights_matrix_rows, int weights_matrix_cols) const override { return new SGD(*this, weights_matrix_rows, weights_matrix_cols); }
};