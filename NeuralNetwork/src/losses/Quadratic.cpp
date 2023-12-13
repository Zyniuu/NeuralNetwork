#include "Losses.h"


namespace nn
{
	namespace loss
	{
		double Quadratic::calcLoss(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			VectorXd error = (true_output - predicted_output).array().square();
			return error.sum();
		}


		VectorXd Quadratic::calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			return (predicted_output - true_output) * 2.0;
		}
	}
}