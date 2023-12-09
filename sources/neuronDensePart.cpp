#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	// Create matrices for weights and bias with random values from the Xavier/Glorot distribution
	double variance = 2.0 / (input_size + output_size);
	m_weights_matrix = getRandomWeights(output_size, input_size, 0.0, variance);
	m_bias_matrix = getRandomBias(output_size, 0.0, variance);
	m_momentum_weights = MatrixXd::Zero(output_size, input_size);
	m_momentum_bias = VectorXd::Zero(output_size);
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


VectorXd NeuronDensePart::backPropagation(VectorXd gradient, double learning_rate)
{
	Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * m_input_matrix.transpose();
	VectorXd gradient_input = m_weights_matrix.transpose() * gradient;

	// Calculate momentum
	m_momentum_weights = (learning_rate * gradient_weights) + (m_momentum_factor * m_momentum_weights);
	m_momentum_bias = (learning_rate * gradient) + (m_momentum_factor * m_momentum_bias);

	// Apply momentum
	m_weights_matrix -= m_momentum_weights;
	m_bias_matrix -= m_momentum_bias;
	return gradient_input;
}