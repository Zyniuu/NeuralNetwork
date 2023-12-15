#include "Losses.h"


namespace nn
{
	namespace loss
	{
		double BinaryCrossEntropy::calcLoss(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			double epsilon = 1e-15;
			double loss = 0.0;

			for (int i = 0; i < true_output.size(); ++i) {
				double p = std::max(std::min(predicted_output[i], 1.0 - epsilon), epsilon);
				loss += -(true_output[i] * std::log(p) + (1.0 - true_output[i]) * std::log(1.0 - p));
			}

			return loss / true_output.size();
		}


		VectorXd BinaryCrossEntropy::calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output)
		{
			return (predicted_output - true_output).array() / (predicted_output.array() * (1.0 - predicted_output.array()));
		}
	}
}
