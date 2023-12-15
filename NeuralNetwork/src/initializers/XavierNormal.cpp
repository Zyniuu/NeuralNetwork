#include "Initializers.h"
#include <chrono>


namespace nn
{
	namespace initializer
	{
		MatrixXd XavierNormal::getRandomWeights()
		{
			double variance = sqrt(6.0 / (m_inputs + m_outputs));
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::normal<MatrixXd>(m_outputs, m_inputs, generator, 0.0, variance);
		}


		VectorXd XavierNormal::getRandomBias()
		{
			double variance = sqrt(6.0 / (m_inputs + m_outputs));
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::normal<VectorXd>(m_outputs, 1, generator, 0.0, variance);
		}
	}
}