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
			return out / true_output.size();
		}


		VectorXd CrossEntropy::calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			VectorXd out = -true_output.array() / (predicted_output.array() + 1e-15);
			//VectorXd out = predicted_output - true_output;
			return out;
		}
	}
}