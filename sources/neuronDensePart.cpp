#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	// Create matrices for weights and bias with random values in range [-1,1]
	weightsMatrix = MatrixXd::Random(output_size, input_size);
	biasMatrix = MatrixXd::Random(output_size, 1);
	// Change the range to [0,1]
	weightsMatrix = (weightsMatrix + MatrixXd::Constant(output_size, input_size, 1)) * 0.5;
	biasMatrix = (biasMatrix + MatrixXd::Constant(output_size, 1, 1)) * 0.5;
}

Matrix<double, Dynamic, 1> NeuronDensePart::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	inputMatrix = inputVals;
	std::cout << weightsMatrix << "\n\n";
	return weightsMatrix * inputMatrix + biasMatrix;
}
