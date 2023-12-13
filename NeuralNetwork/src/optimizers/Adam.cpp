#include "Optimizers.h"


namespace nn
{
	namespace optimizer
	{
		Adam::Adam(const double& learning_rate, const double& beta1, const double& beta2, const double& epsilon)
			: Optimizer(learning_rate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}


		Adam::Adam(const Adam& other, const int& weights_matrix_rows, const int& weights_matrix_cols) : 
			Optimizer(other), 
			m_beta1(other.m_beta1), 
			m_beta2(other.m_beta2), 
			m_epsilon(other.m_epsilon), 
			m_weights_first_momentum(other.m_weights_first_momentum), 
			m_weights_second_momentum(other.m_weights_second_momentum), 
			m_bias_first_momentum(other.m_bias_first_momentum), 
			m_bias_second_momentum(other.m_bias_second_momentum)
		{
			m_weights_first_momentum = MatrixXd::Zero(weights_matrix_rows, weights_matrix_cols);
			m_weights_second_momentum = MatrixXd::Zero(weights_matrix_rows, weights_matrix_cols);
			m_bias_first_momentum = VectorXd::Zero(weights_matrix_rows);
			m_bias_second_momentum = VectorXd::Zero(weights_matrix_rows);
		}


		Matrix<double, Dynamic, Dynamic> Adam::getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient)
		{
			m_weights_first_momentum = (m_beta1 * m_weights_first_momentum).colwise() + ((1 - m_beta1) * gradient);
			m_weights_second_momentum = (m_beta2 * m_weights_second_momentum).colwise() + ((1 - m_beta2) * gradient.cwiseProduct(gradient));
			MatrixXd m_weights_first_momentum_hat = m_weights_first_momentum / (1 - std::pow(m_beta1, m_t_weights));
			MatrixXd m_weights_second_momentum_hat = m_weights_second_momentum / (1 - std::pow(m_beta2, m_t_weights));
			m_t_weights += 1;
			return (m_learning_rate * m_weights_first_momentum_hat.array() / (m_weights_second_momentum_hat.array().sqrt() + m_epsilon)).matrix() * input_vector.transpose();
		}


		VectorXd Adam::getDeltaBias(const VectorXd& gradient)
		{
			m_bias_first_momentum = (m_beta1 * m_bias_first_momentum) + ((1 - m_beta1) * gradient);
			m_bias_second_momentum = (m_beta2 * m_bias_second_momentum) + ((1 - m_beta2) * gradient.cwiseProduct(gradient));
			VectorXd m_bias_first_momentum_hat = m_bias_first_momentum / (1 - std::pow(m_beta1, m_t_bias));
			VectorXd m_bias_second_momentum_hat = m_bias_second_momentum / (1 - std::pow(m_beta2, m_t_bias));
			m_t_bias += 1;
			return m_learning_rate * m_bias_first_momentum_hat.array() / (m_bias_second_momentum_hat.array().sqrt() + m_epsilon);
		}
	}
}