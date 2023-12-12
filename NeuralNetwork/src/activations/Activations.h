#pragma once
#include "../layers/Layers.h"


namespace nn
{
	namespace activation
	{
		enum Activations { TANH, RELU, SIGMOID, SOFTMAX };

		class ReLUActivation : public layer::NeuronActivationPart
		{
		public:
			ReLUActivation() : NeuronActivationPart(&ReLUActivation::reluFunc, &ReLUActivation::reluFuncPrime) {}
			static VectorXd reluFunc(VectorXd input_vals);
			static VectorXd reluFuncPrime(VectorXd input_vals);
		};


		class SigmoidActivation : public layer::NeuronActivationPart
		{
		public:
			SigmoidActivation() : NeuronActivationPart(&SigmoidActivation::sigmoidFunc, &SigmoidActivation::sigmoidFuncPrime) {}
			static VectorXd sigmoidFunc(VectorXd input_vals);
			static VectorXd sigmoidFuncPrime(VectorXd input_vals);
		};


		class TanhActivation : public layer::NeuronActivationPart
		{
		public:
			TanhActivation() : NeuronActivationPart(&TanhActivation::tanhFunc, &TanhActivation::tanhFuncPrime) {}
			static VectorXd tanhFunc(VectorXd input_vals);
			static VectorXd tanhFuncPrime(VectorXd input_vals);
		};


		class SoftmaxActivation : public layer::Layer
		{
		private:
			VectorXd m_output_matrix;

		public:
			VectorXd feedForward(VectorXd input_vals);
			VectorXd backPropagation(VectorXd gradient, const optimizer::Optimizer& optimizer);
			int getType() const { return layer::ACTIVATION; }
		};
	}
}
