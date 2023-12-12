#include "Optimizers.h"


namespace nn
{
	namespace optimizer
	{
		SGD::SGD(const double& learning_rate, const double& momentum)
			: Optimizer(learning_rate), m_momentum(momentum) {}


		SGD::SGD(const SGD& other, const int& weights_matrix_rows, const int& weights_matrix_cols)
			: Optimizer(other), m_momentum(other.m_momentum), m_momentum_weights(other.m_momentum_weights), m_momentum_bias(other.m_momentum_bias)
		{
			m_momentum_weights = MatrixXd::Zero(weights_matrix_rows, weights_matrix_cols);
			m_momentum_bias = VectorXd::Zero(weights_matrix_rows);
		}


		Matrix<double, Dynamic, Dynamic> SGD::getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient)
		{
			Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * input_vector.transpose();
			m_momentum_weights = (m_learning_rate * gradient_weights) + (m_momentum * m_momentum_weights);
			return m_momentum_weights;
		}


		VectorXd SGD::getDeltaBias(const VectorXd& gradient)
		{
			m_momentum_bias = (m_learning_rate * gradient) + (m_learning_rate * m_momentum_bias);
			return m_momentum_bias;
		}
	}
}
