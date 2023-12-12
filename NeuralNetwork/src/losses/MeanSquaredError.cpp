#include "Losses.h"


namespace nn
{
	namespace loss
	{
		double MeanSquaredError::calcLoss(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			VectorXd error = (true_output - predicted_output).array().square();
			return error.mean();
		}


		VectorXd MeanSquaredError::calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			double scale = 2.0 / true_output.size();
			return scale * (predicted_output - true_output);
		}
	}
}
