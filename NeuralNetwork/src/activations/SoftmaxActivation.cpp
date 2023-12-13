#include "Activations.h"


namespace nn
{
	namespace activation
	{
		VectorXd SoftmaxActivation::feedForward(const VectorXd& input_vals)
		{
			VectorXd exp_vals = (input_vals.array() - input_vals.maxCoeff()).exp();
			m_output_matrix = exp_vals / exp_vals.sum();
			return m_output_matrix;
		}


		VectorXd SoftmaxActivation::backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer)
		{
			VectorXd d_softmax = VectorXd::Zero(m_output_matrix.size());
			for (unsigned i = 0; i < m_output_matrix.size(); ++i) 
			{
				for (unsigned j = 0; j < m_output_matrix.size(); ++j) 
				{
					if (i == j)
						d_softmax(i) += m_output_matrix(i) * (1 - m_output_matrix(i)) * gradient(i);
					else
						d_softmax(i) -= m_output_matrix(i) * m_output_matrix(j) * gradient(j);
				}
			}
			return d_softmax;
		}
	}
}
