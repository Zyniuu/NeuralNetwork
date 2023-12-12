#pragma once
#include "../../assets/EigenRand/EigenRand"


using namespace Eigen;

namespace nn
{
	namespace loss
	{
		enum { MSE };

		class Loss
		{
		public:
			virtual double calcLoss(const VectorXd& true_output, const VectorXd& predicted_output) = 0;
			virtual VectorXd calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output) = 0;
		};


		class MeanSquaredError : public Loss
		{
		public:
			double calcLoss(const VectorXd& true_output, const VectorXd& predicted_output);
			VectorXd calcLossPrime(const VectorXd& true_output, const VectorXd& predicted_output);
		};
	}
}