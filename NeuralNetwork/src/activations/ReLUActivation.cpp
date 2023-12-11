#include "Activations.h"


namespace nn
{
	VectorXd ReLUActivation::reluFunc(VectorXd input_vals)
	{
		input_vals = input_vals.array().unaryExpr([](double element) { return (element > 0.0) ? element : 0.0; });
		return input_vals;
	}


	VectorXd ReLUActivation::reluFuncPrime(VectorXd input_vals)
	{
		input_vals = input_vals.array().unaryExpr([](double element) { return (element > 0.0) ? 1.0 : 0.0; });
		return input_vals;
	}
}
