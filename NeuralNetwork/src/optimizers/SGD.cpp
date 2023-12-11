#include "Optimizers.h"


namespace nn
{
	SGD::SGD(double learning_rate, double momentum)
		: Optimizer(learning_rate), m_momentum(momentum) {}


	SGD::SGD(const SGD& other, int weights_matrix_rows, int weights_matrix_cols)
		: Optimizer(other), m_momentum(other.m_momentum), m_momentum_weights(other.m_momentum_weights), m_momentum_bias(other.m_momentum_bias)
	{
		m_momentum_weights = MatrixXd::Zero(weights_matrix_rows, weights_matrix_cols);
		m_momentum_bias = VectorXd::Zero(weights_matrix_rows);
	}


	Matrix<double, Dynamic, Dynamic> SGD::getDeltaWeights(VectorXd input_vector, VectorXd gradient)
	{
		Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * input_vector.transpose();
		m_momentum_weights = (m_learning_rate * gradient_weights) + (m_momentum * m_momentum_weights);
		return m_momentum_weights;
	}


	VectorXd SGD::getDeltaBias(VectorXd gradient)
	{
		m_momentum_bias = (m_learning_rate * gradient) + (m_learning_rate * m_momentum_bias);
		return m_momentum_bias;
	}
}
