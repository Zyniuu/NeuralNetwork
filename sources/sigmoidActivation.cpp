#include "../headers/sigmoidActivation.h"


Matrix<double, Dynamic, 1> SigmoidActivation::sigmoidFunc(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = (-inputVals).array().exp();
	inputVals = inputVals.array().unaryExpr([](double element) { return 1 / (1 + element); });
	return inputVals;
}

Matrix<double, Dynamic, 1> SigmoidActivation::sigmoidFuncPrime(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = sigmoidFunc(inputVals);
	inputVals = inputVals.array().unaryExpr([](double element) { return element * (1 - element); });
	return inputVals;
}
