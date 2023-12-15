#include "Initializers.h"
#include <chrono>


namespace nn
{
	namespace initializer
	{
		MatrixXd XavierUniform::getRandomWeights()
		{
			double variance = sqrt(6.0 / (m_inputs + m_outputs));
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::uniformReal<MatrixXd>(m_outputs, m_inputs, generator, 0.0, variance);
		}


		VectorXd XavierUniform::getRandomBias()
		{
			double variance = sqrt(6.0 / (m_inputs + m_outputs));
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::uniformReal<VectorXd>(m_outputs, 1, generator, 0.0, variance);
		}
	}
}