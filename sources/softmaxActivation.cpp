#include "../headers/SoftmaxActivation.h"


VectorXd SoftmaxActivation::feedForward(VectorXd inputVals)
{
	VectorXd expVals = inputVals.array().exp();
	outputMatrix = expVals / expVals.sum();
	return outputMatrix;
}

VectorXd SoftmaxActivation::backPropagation(VectorXd gradient, double learning_rate)
{
	/*
	VectorXd softmax_result = outputMatrix;
	Matrix<double, Dynamic, 1> out = (softmax_result.array() * (1 - softmax_result.array()));
	return out;
	*/
	int n = outputMatrix.size();
	Matrix<double, Dynamic, Dynamic> identity = MatrixXd::Identity(n, n);
	return (identity - outputMatrix.transpose()).cwiseProduct(outputMatrix) * gradient;
}