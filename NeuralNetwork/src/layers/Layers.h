#pragma once
#include "../optimizers/Optimizers.h"
#include "../../assets/rapidxml/rapidxml.hpp"
#include "../../assets/rapidxml/rapidxml_print.hpp"
#include "../initializers/Initializers.h"


namespace nn
{
	namespace layer
	{
		enum type { DENSE, ACTIVATION };

		class Layer
		{
		protected:
			VectorXd m_input_matrix;

		public:
			virtual VectorXd feedForward(const VectorXd& input_vals) = 0;
			virtual VectorXd backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer) = 0;
			virtual int getType() const = 0;
		};


		class NeuronDensePart : public Layer
		{
		private:
			optimizer::Optimizer* m_optimizer;
			initializer::Initializer* m_initializer;

		protected:
			Matrix<double, Dynamic, Dynamic> m_weights_matrix;
			VectorXd m_bias_matrix;

		public:
			NeuronDensePart(const int& input_size, const int& output_size, const int& initializer);
			NeuronDensePart(const int& input_size, const int& output_size, rapidxml::xml_node<>* layer_node);

			VectorXd feedForward(const VectorXd& input_vals);
			VectorXd backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer);
			int getType() const { return DENSE; }
			void saveLayer(rapidxml::xml_document<>* document, rapidxml::xml_node<>* layer_node);

		private:
			static initializer::Initializer* generateInitializer(const int& initializer, const int& input_size, const int& output_size);
			static VectorXd getNodeValues(rapidxml::xml_node<>* node, const int& size);
			static MatrixXd getMatrixFromVector(const int& rows, const int& cols, const VectorXd& data);
		};


		class NeuronActivationPart : public Layer
		{
		protected:
			VectorXd(*m_activation_func)(VectorXd);
			VectorXd(*m_activation_func_prime)(VectorXd);

			NeuronActivationPart(VectorXd(*activation_func)(VectorXd), VectorXd(*activation_func_prime)(VectorXd));
			VectorXd feedForward(const VectorXd& input_vals);
			VectorXd backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer);
			int getType() const { return ACTIVATION; }
		};
	}
}
