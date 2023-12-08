#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	Rand::Vmt19937_64 generator;
	// Create matrices for weights and bias with random values from the normal distribution
	weightsMatrix = Rand::normal<MatrixXd>(output_size, input_size, generator, 0.0, 1.0);
	biasMatrix = Rand::normal<MatrixXd>(output_size, 1, generator, 0.0, 1.0);
	// Create matrices for weights and bias with random values in range [-1,1]
	// weightsMatrix = MatrixXd::Random(output_size, input_size);
	// biasMatrix = MatrixXd::Random(output_size, 1);
}

Matrix<double, Dynamic, 1> NeuronDensePart::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	inputMatrix = inputVals;
	return weightsMatrix * inputMatrix + biasMatrix;
}

Matrix<double, Dynamic, 1> NeuronDensePart::backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate)
{
	Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * inputMatrix.transpose();
	Matrix<double, Dynamic, 1> gradient_input = weightsMatrix.transpose() * gradient;
	weightsMatrix -= learning_rate * gradient_weights;
	biasMatrix -= learning_rate * gradient;
	return gradient_input;
}