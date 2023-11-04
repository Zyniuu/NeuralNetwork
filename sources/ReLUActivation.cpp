#include "../headers/ReLUActivation.h"


Matrix<double, Dynamic, 1> ReLUActivation::reluFunc(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return (element > 0.0) ? element : 0.0; });
	return inputVals;
}

Matrix<double, Dynamic, 1> ReLUActivation::reluFuncPrime(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return (element > 0.0) ? 1.0 : 0.0; });
	return inputVals;
}