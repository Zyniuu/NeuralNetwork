#include "Optimizers.h"


namespace nn
{
	namespace optimizer
	{
		AdaGrad::AdaGrad(const double& learning_rate, const double& initial_accumulator, const double& epsilon)
			: Optimizer(learning_rate), m_initial_accumulator(initial_accumulator), m_epsilon(epsilon) {}


		AdaGrad::AdaGrad(const AdaGrad& other, const int& weights_matrix_rows, const int& weights_matrix_cols) : 
			Optimizer(other), 
			m_initial_accumulator(other.m_initial_accumulator), 
			m_epsilon(other.m_epsilon), 
			m_accumulated_weights(other.m_accumulated_weights), 
			m_accumulated_bias(other.m_accumulated_bias)
		{
			m_accumulated_weights = MatrixXd::Constant(weights_matrix_rows, weights_matrix_cols, m_initial_accumulator);
			m_accumulated_bias = VectorXd::Constant(weights_matrix_rows, m_initial_accumulator);
		}


		Matrix<double, Dynamic, Dynamic> AdaGrad::getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient)
		{
			m_accumulated_weights = m_accumulated_weights.colwise() + gradient.cwiseProduct(gradient);
			return (m_learning_rate * gradient * input_vector.transpose()).array() / (m_accumulated_weights.array().sqrt() + m_epsilon);
		}


		VectorXd AdaGrad::getDeltaBias(const VectorXd& gradient)
		{
			m_accumulated_bias += gradient.cwiseProduct(gradient);
			return (m_learning_rate * gradient).array() / (m_accumulated_bias.array().sqrt() + m_epsilon);
		}
	}
}