#include "../headers/tanhActivation.h"


VectorXd TanhActivation::tanhFunc(VectorXd inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return std::tanh(element); });
	return inputVals;
}

VectorXd TanhActivation::tanhFuncPrime(VectorXd inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return 1 - std::pow(std::tanh(element), 2); });
	return inputVals;
}
