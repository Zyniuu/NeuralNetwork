#include "Activations.h"


namespace nn
{
	VectorXd TanhActivation::tanhFunc(VectorXd input_vals)
	{
		input_vals = input_vals.array().unaryExpr([](double element) { return std::tanh(element); });
		return input_vals;
	}


	VectorXd TanhActivation::tanhFuncPrime(VectorXd input_vals)
	{
		input_vals = input_vals.array().unaryExpr([](double element) { return 1 - std::pow(std::tanh(element), 2); });
		return input_vals;
	}
}
