#pragma once
#include "optimizers.h"


namespace nn
{
	class Layer
	{
	protected:
		VectorXd m_input_matrix;

	public:
		virtual VectorXd feedForward(VectorXd input_vals) = 0;
		virtual VectorXd backPropagation(VectorXd gradient, const Optimizer& optimizer) = 0;
		virtual bool isNeuronDensePart() const { return false; }
	};


	class NeuronDensePart : public Layer
	{
	private:
		Optimizer* m_optimizer;

		Matrix<double, Dynamic, Dynamic> getRandomWeights(int rows, int cols, double mean, double variance);
		VectorXd getRandomBias(int size, double mean, double variance);

	protected:
		Matrix<double, Dynamic, Dynamic> m_weights_matrix;
		VectorXd m_bias_matrix;

	public:
		NeuronDensePart(int input_size, int output_size);
		NeuronDensePart(Matrix<double, Dynamic, Dynamic> weights_matrix, VectorXd bias_matrix);
		VectorXd feedForward(VectorXd input_vals);
		VectorXd backPropagation(VectorXd gradient, const Optimizer& optimizer);
		Matrix<double, Dynamic, Dynamic> getWeightsMatrix() const { return m_weights_matrix; };
		VectorXd getBiasMatrix() const { return m_bias_matrix; };
		bool isNeuronDensePart() const override { return true; }
	};


	class NeuronActivationPart : public Layer
	{
	protected:
		VectorXd(*m_activation_func)(VectorXd);
		VectorXd(*m_activation_func_prime)(VectorXd);

		NeuronActivationPart(VectorXd(*activation_func)(VectorXd), VectorXd(*activation_func_prime)(VectorXd));
		VectorXd feedForward(VectorXd input_vals);
		VectorXd backPropagation(VectorXd gradient, const Optimizer& optimizer);
	};
}
