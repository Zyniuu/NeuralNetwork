#pragma once
#include "layer.h"


class NeuronDensePart : public Layer
{
private:
	double m_momentum_factor = 0.9;
	Matrix<double, Dynamic, Dynamic> m_momentum_weights;
	VectorXd m_momentum_bias;

	Matrix<double, Dynamic, Dynamic> getRandomWeights(int rows, int cols, double mean, double variance);
	VectorXd getRandomBias(int size, double mean, double variance);

protected:
	Matrix<double, Dynamic, Dynamic> m_weights_matrix;
	VectorXd m_bias_matrix;

public:
	NeuronDensePart(int input_size, int output_size);
	NeuronDensePart(Matrix<double, Dynamic, Dynamic> weights_matrix, VectorXd bias_matrix);
	VectorXd feedForward(VectorXd input_vals);
	VectorXd backPropagation(VectorXd gradient, double learning_rate);
	Matrix<double, Dynamic, Dynamic> getWeightsMatrix() const { return m_weights_matrix; };
	VectorXd getBiasMatrix() const { return m_bias_matrix; };
	bool isNeuronDensePart() const override { return true; }
};
