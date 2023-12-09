#include "../headers/sigmoidActivation.h"


VectorXd SigmoidActivation::sigmoidFunc(VectorXd input_vals)
{
	input_vals = (-input_vals).array().exp();
	input_vals = input_vals.array().unaryExpr([](double element) { return 1 / (1 + element); });
	return input_vals;
}


VectorXd SigmoidActivation::sigmoidFuncPrime(VectorXd input_vals)
{
	input_vals = sigmoidFunc(input_vals);
	input_vals = input_vals.array().unaryExpr([](double element) { return element * (1 - element); });
	return input_vals;
}
