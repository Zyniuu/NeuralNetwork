#include "../headers/SoftmaxActivation.h"


Matrix<double, Dynamic, 1> SoftmaxActivation::softmaxFunc(Matrix<double, Dynamic, 1> inputVals)
{
	Matrix<double, Dynamic, 1> expVals = inputVals.array().exp();
	return expVals / (expVals.sum() + 1e-8);
}

Matrix<double, Dynamic, 1> SoftmaxActivation::softmaxPrime(Matrix<double, Dynamic, 1> inputVals)
{
	VectorXd softmax_result = softmaxFunc(inputVals);
	return (softmax_result.array() * (1 - softmax_result.array()));
}