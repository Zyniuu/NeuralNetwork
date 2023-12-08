#include "../headers/sigmoidActivation.h"


VectorXd SigmoidActivation::sigmoidFunc(VectorXd inputVals)
{
	inputVals = (-inputVals).array().exp();
	inputVals = inputVals.array().unaryExpr([](double element) { return 1 / (1 + element); });
	return inputVals;
}

VectorXd SigmoidActivation::sigmoidFuncPrime(VectorXd inputVals)
{
	inputVals = sigmoidFunc(inputVals);
	inputVals = inputVals.array().unaryExpr([](double element) { return element * (1 - element); });
	return inputVals;
}
