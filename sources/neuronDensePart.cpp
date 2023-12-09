#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	// Create matrices for weights and bias with random values from the Xavier/Glorot distribution
	double variance = 2.0 / (input_size + output_size);
	weightsMatrix = getRandomWeights(output_size, input_size, 0.0, variance);
	biasMatrix = getRandomBias(output_size, 0.0, variance);
	momentumWeights = MatrixXd::Zero(output_size, input_size);
	momentumBias = VectorXd::Zero(output_size);
	// Create matrices for weights and bias with random values in range [-1,1]
	// weightsMatrix = MatrixXd::Random(output_size, input_size);
	// biasMatrix = MatrixXd::Random(output_size, 1);
}

NeuronDensePart::NeuronDensePart(Matrix<double, Dynamic, Dynamic> _weightsMatrix, VectorXd _biasMatrix)
{
	weightsMatrix = _weightsMatrix;
	biasMatrix = _biasMatrix;
}

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

VectorXd NeuronDensePart::feedForward(VectorXd inputVals)
{
	inputMatrix = inputVals;
	return weightsMatrix * inputMatrix + biasMatrix;
}

VectorXd NeuronDensePart::backPropagation(VectorXd gradient, double learning_rate)
{
	Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * inputMatrix.transpose();
	VectorXd gradient_input = weightsMatrix.transpose() * gradient;

	// Calculate momentum
	momentumWeights = (learning_rate * gradient_weights) + (momentumFactor * momentumWeights);
	momentumBias = (learning_rate * gradient) + (momentumFactor * momentumBias);

	// Apply momentum
	weightsMatrix -= momentumWeights;
	biasMatrix -= momentumBias;
	return gradient_input;
}