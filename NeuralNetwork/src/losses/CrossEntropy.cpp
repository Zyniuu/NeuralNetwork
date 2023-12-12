#include "Losses.h"


namespace nn
{
	namespace loss
	{
		double CrossEntropy::calcLoss(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			double out = 0.0;
			for (unsigned i = 0; i < true_output.size(); ++i)
				out += -(true_output(i) * log(predicted_output(i) + 1e-15));
			return out;
		}


		VectorXd CrossEntropy::calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			//VectorXd out(true_output.size());
			//for (unsigned i = 0; i < true_output.size(); ++i)
			//	out(i) = (predicted_output(i) - true_output(i)) / (predicted_output(i) * (1.0 - predicted_output(i)) + 1e-15);
			//return out;
			return predicted_output - true_output;
		}
	}
}