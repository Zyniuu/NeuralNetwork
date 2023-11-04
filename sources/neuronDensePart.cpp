#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	// Create matrices for weights and bias with random values in range [-1,1]
	weightsMatrix = MatrixXd::Random(output_size, input_size);
	biasMatrix = MatrixXd::Random(output_size, 1);
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