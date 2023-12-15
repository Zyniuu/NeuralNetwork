#pragma once
#include "../../assets/EigenRand/EigenRand"


using namespace Eigen;

namespace nn
{
	namespace initializer
	{
		enum { HE_NORMAL, HE_UNIFORM, XAVIER_NORMAL, XAVIER_UNIFORM };

		class Initializer
		{
		protected:
			int m_inputs;
			int m_outputs;

		public:
			Initializer(const int& inputs, const int& outputs) : m_inputs(inputs), m_outputs(outputs) {}
			virtual MatrixXd getRandomWeights() = 0;
			virtual VectorXd getRandomBias() = 0;
		};


		class HeNormal : public Initializer
		{
		public:
			HeNormal(const int& inputs, const int& outputs) : Initializer(inputs, outputs) {}
			MatrixXd getRandomWeights();
			VectorXd getRandomBias();
		};


		class HeUniform : public Initializer
		{
		public:
			HeUniform(const int& inputs, const int& outputs) : Initializer(inputs, outputs) {}
			MatrixXd getRandomWeights();
			VectorXd getRandomBias();
		};


		class XavierNormal : public Initializer
		{
		public:
			XavierNormal(const int& inputs, const int& outputs) : Initializer(inputs, outputs) {}
			MatrixXd getRandomWeights();
			VectorXd getRandomBias();
		};


		class XavierUniform : public Initializer
		{
		public:
			XavierUniform(const int& inputs, const int& outputs) : Initializer(inputs, outputs) {}
			MatrixXd getRandomWeights();
			VectorXd getRandomBias();
		};
	}
}