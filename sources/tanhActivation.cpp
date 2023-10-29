#include "../headers/tanhActivation.h"


Matrix<double, Dynamic, 1> TanhActivation::tanhFunc(Matrix<double, Dynamic, 1> inputVals)
{
	inputVals = inputVals.array().unaryExpr([](double element) { return std::tanh(element); });
	std::cout << inputVals << "\n\n";
	return inputVals;
}
