#include "../headers/ReLUActivation.h"


VectorXd ReLUActivation::reluFunc(VectorXd inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return (element > 0.0) ? element : 0.0; });
	return inputVals;
}

VectorXd ReLUActivation::reluFuncPrime(VectorXd inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return (element > 0.0) ? 1.0 : 0.0; });
	return inputVals;
}