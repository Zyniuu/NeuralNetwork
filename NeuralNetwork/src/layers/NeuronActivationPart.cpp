#include "Layers.h"


namespace nn
{
	namespace layer
	{
		NeuronActivationPart::NeuronActivationPart(VectorXd(*activation_func)(VectorXd), VectorXd(*activation_func_prime)(VectorXd))
			: m_activation_func(activation_func), m_activation_func_prime(activation_func_prime) {}


		VectorXd NeuronActivationPart::feedForward(const VectorXd& inputVals)
		{
			m_input_matrix = inputVals;
			return m_activation_func(m_input_matrix);
		}


		VectorXd NeuronActivationPart::backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer)
		{
			return gradient.cwiseProduct(m_activation_func_prime(m_input_matrix));
		}
	}
}
