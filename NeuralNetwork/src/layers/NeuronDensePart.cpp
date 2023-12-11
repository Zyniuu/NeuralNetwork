#include "Layers.h"


namespace nn
{
	NeuronDensePart::NeuronDensePart(int input_size, int output_size)
	{
		// Create matrices for weights and bias with random values from the Xavier/Glorot distribution
		double variance = 2.0 / (input_size + output_size);
		m_weights_matrix = getRandomWeights(output_size, input_size, 0.0, variance);
		m_bias_matrix = getRandomBias(output_size, 0.0, variance);
	}


	NeuronDensePart::NeuronDensePart(Matrix<double, Dynamic, Dynamic> weights_matrix, VectorXd bias_matrix)
		: m_weights_matrix(weights_matrix), m_bias_matrix(bias_matrix) {}


	Matrix<double, Dynamic, Dynamic> NeuronDensePart::getRandomWeights(int rows, int cols, double mean, double variance)
	{
		Rand::Vmt19937_64 generator;
		return Rand::normal<MatrixXd>(rows, cols, generator, 0.0, variance);
	}


	VectorXd NeuronDensePart::getRandomBias(int size, double mean, double variance)
	{
		Rand::Vmt19937_64 generator;
		return Rand::normal<VectorXd>(size, 1, generator, 0.0, variance);
	}


	VectorXd NeuronDensePart::feedForward(VectorXd input_vals)
	{
		m_input_matrix = input_vals;
		return m_weights_matrix * m_input_matrix + m_bias_matrix;
	}


	VectorXd NeuronDensePart::backPropagation(VectorXd gradient, const Optimizer& optimizer)
	{
		if (!m_optimizer)
			m_optimizer = optimizer.clone(m_weights_matrix.rows(), m_weights_matrix.cols());

		VectorXd gradient_input = m_weights_matrix.transpose() * gradient;
		m_weights_matrix -= m_optimizer->getDeltaWeights(m_input_matrix, gradient);
		m_bias_matrix -= m_optimizer->getDeltaBias(gradient);
		return gradient_input;
	}
}
