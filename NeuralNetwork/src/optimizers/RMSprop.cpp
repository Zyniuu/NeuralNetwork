#include "Optimizers.h"


namespace nn
{
	namespace optimizer
	{
		RMSprop::RMSprop(const double& learning_rate, const double& decay_factor, const double& epsilon)
			: Optimizer(learning_rate), m_decay_factor(decay_factor), m_epsilon(epsilon) {}


		RMSprop::RMSprop(const RMSprop& other, const int& weights_matrix_rows, const int& weights_matrix_cols) : 
			Optimizer(other), 
			m_decay_factor(other.m_decay_factor),
			m_epsilon(other.m_epsilon), 
			m_accumulated_weights(other.m_accumulated_weights), 
			m_accumulated_bias(other.m_accumulated_bias) 
		{
			m_accumulated_weights = MatrixXd::Zero(weights_matrix_rows, weights_matrix_cols);
			m_accumulated_bias = VectorXd::Zero(weights_matrix_rows);
		}


		Matrix<double, Dynamic, Dynamic> RMSprop::getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient)
		{
			m_accumulated_weights = (m_decay_factor * m_accumulated_weights.array()).colwise() + (1.0 - m_decay_factor) * gradient.array().square();
			return (m_learning_rate * gradient * input_vector.transpose()).array() / (m_accumulated_weights.array().sqrt() + m_epsilon);
		}


		VectorXd RMSprop::getDeltaBias(const VectorXd& gradient)
		{
			m_accumulated_bias = m_decay_factor * m_accumulated_bias.array() + (1.0 - m_decay_factor) * gradient.array().square();
			return (m_learning_rate * gradient).array() / (m_accumulated_bias.array().sqrt() + m_epsilon);
		}
	}
}