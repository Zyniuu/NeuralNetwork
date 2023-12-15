#include "Initializers.h"
#include <chrono>


namespace nn
{
	namespace initializer
	{
		MatrixXd HeNormal::getRandomWeights()
		{
			double variance = sqrt(2.0 / m_inputs);
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::normal<MatrixXd>(m_outputs, m_inputs, generator, 0.0, variance);
		}


		VectorXd HeNormal::getRandomBias()
		{
			double variance = sqrt(2.0 / m_inputs);
			Rand::Vmt19937_64 generator{ (unsigned)std::chrono::system_clock::now().time_since_epoch().count() };
			return Rand::normal<VectorXd>(m_outputs, 1, generator, 0.0, variance);
		}
	}
}