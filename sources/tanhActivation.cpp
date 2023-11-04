#include "../headers/tanhActivation.h"


Matrix<double, Dynamic, 1> TanhActivation::tanhFunc(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return std::tanh(element); });
	return inputVals;
}

Matrix<double, Dynamic, 1> TanhActivation::tanhFuncPrime(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return 1 - std::pow(std::tanh(element), 2); });
	return inputVals;
}
