#include "Activations.h"


namespace nn
{
	namespace activation
	{
		VectorXd SigmoidActivation::sigmoidFunc(VectorXd input_vals)
		{
			/*
			input_vals = (-input_vals).array().exp();
			input_vals = input_vals.array().unaryExpr([](double element) { return 1 / (1 + element); });
			return input_vals;
			*/
			double maxExp = 20.0;

			for (int i = 0; i < input_vals.size(); ++i)
				input_vals[i] = std::exp(std::max(-maxExp, std::min(maxExp, -input_vals[i])));
			return (1.0 + input_vals.array()).inverse();
		}


		VectorXd SigmoidActivation::sigmoidFuncPrime(VectorXd input_vals)
		{
			input_vals = sigmoidFunc(input_vals);
			//input_vals = input_vals.array().unaryExpr([](double element) { return element * (1 - element); });
			return input_vals.array() * (1 - input_vals.array());
		}
	}
}
