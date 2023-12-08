#include "../headers/SoftmaxActivation.h"


Matrix<double, Dynamic, 1> SoftmaxActivation::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	Matrix<double, Dynamic, 1> expVals = inputVals.array().exp();
	outputMatrix = expVals / expVals.sum();
	return outputMatrix;
}

Matrix<double, Dynamic, 1> SoftmaxActivation::backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate)
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