#include "../headers/SoftmaxActivation.h"


VectorXd SoftmaxActivation::feedForward(VectorXd input_vals)
{
	VectorXd exp_vals = input_vals.array().exp();
	m_output_matrix = exp_vals / exp_vals.sum();
	return m_output_matrix;
}


VectorXd SoftmaxActivation::backPropagation(VectorXd gradient, double learning_rate)
{
	int n = m_output_matrix.size();
	Matrix<double, Dynamic, Dynamic> identity = MatrixXd::Identity(n, n);
	return (identity - m_output_matrix.transpose()).cwiseProduct(m_output_matrix) * gradient;
}