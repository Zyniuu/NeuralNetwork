#pragma once
#include "layer.h"


class NeuronDensePart : public Layer
{
private:
	Matrix<double, Dynamic, Dynamic> getRandomWeights(int rows, int cols, double mean, double variance);
	VectorXd getRandomBias(int size, double mean, double variance);
protected:
	Matrix<double, Dynamic, Dynamic> weightsMatrix;
	VectorXd biasMatrix;
public:
	NeuronDensePart(int input_size, int output_size);
	NeuronDensePart(Matrix<double, Dynamic, Dynamic> _weightsMatrix, VectorXd _biasMatrix);
	VectorXd feedForward(VectorXd inputVals);
	VectorXd backPropagation(VectorXd gradient, double learning_rate);
	Matrix<double, Dynamic, Dynamic> getWeightsMatrix() const { return weightsMatrix; };
	VectorXd getBiasMatrix() const { return biasMatrix; };
	bool isNeuronDensePart() const override { return true; }
};
