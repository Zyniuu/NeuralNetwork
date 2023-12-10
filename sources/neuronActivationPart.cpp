#include "../headers/neuronActivationPart.h"


NeuronActivationPart::NeuronActivationPart(VectorXd (*activation_func)(VectorXd), VectorXd (*activation_func_prime)(VectorXd))
	: m_activation_func(activation_func), m_activation_func_prime(activation_func_prime){}


VectorXd NeuronActivationPart::feedForward(VectorXd inputVals)
{
	m_input_matrix = inputVals;
	return m_activation_func(m_input_matrix);
}


VectorXd NeuronActivationPart::backPropagation(VectorXd gradient, const Optimizer& optimizer)
{
	return gradient.cwiseProduct(m_activation_func_prime(m_input_matrix));
}