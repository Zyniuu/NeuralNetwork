#include "Layers.h"
#include <chrono>


namespace nn
{
	namespace layer
	{
		NeuronDensePart::NeuronDensePart(const int& input_size, const int& output_size, const int& initializer)
		{
			m_initializer = generateInitializer(initializer, input_size, output_size);
			m_weights_matrix = m_initializer->getRandomWeights();
			m_bias_matrix = m_initializer->getRandomBias();
		}


		NeuronDensePart::NeuronDensePart(const int& input_size, const int& output_size, rapidxml::xml_node<>* layer_node)
		{
			rapidxml::xml_node<>* weights = layer_node->first_node("Weights");
			rapidxml::xml_node<>* bias = layer_node->first_node("Bias");

			m_weights_matrix = getMatrixFromVector(output_size, input_size, getNodeValues(weights, output_size * input_size));
			m_bias_matrix = getNodeValues(bias, output_size);
		}


		initializer::Initializer* NeuronDensePart::generateInitializer(const int& initializer, const int& input_size, const int& output_size)
		{
			switch (initializer)
			{
			case initializer::HE_NORMAL:
				return new initializer::HeNormal(input_size, output_size);
			case initializer::HE_UNIFORM:
				return new initializer::HeUniform(input_size, output_size);
			case initializer::XAVIER_NORMAL:
				return new initializer::XavierNormal(input_size, output_size);
			case initializer::XAVIER_UNIFORM:
				return new initializer::XavierUniform(input_size, output_size);
			default:
				return nullptr;
			}
		}


		VectorXd NeuronDensePart::feedForward(const VectorXd& input_vals)
		{
			m_input_matrix = input_vals;
			return m_weights_matrix * m_input_matrix + m_bias_matrix;
		}


		VectorXd NeuronDensePart::backPropagation(const VectorXd& gradient, const optimizer::Optimizer& optimizer)
		{
			if (!m_optimizer)
				m_optimizer = optimizer.clone(m_weights_matrix.rows(), m_weights_matrix.cols());

			VectorXd gradient_input = m_weights_matrix.transpose() * gradient;
			m_weights_matrix -= m_optimizer->getDeltaWeights(m_input_matrix, gradient);
			m_bias_matrix -= m_optimizer->getDeltaBias(gradient);
			return gradient_input;
		}


		void NeuronDensePart::saveLayer(rapidxml::xml_document<>* document, rapidxml::xml_node<>* layer_node)
		{
			rapidxml::xml_node<>* weights = document->allocate_node(rapidxml::node_element, "Weights");
			rapidxml::xml_node<>* bias = document->allocate_node(rapidxml::node_element, "Bias");

			for (double x : m_weights_matrix.reshaped())
			{
				rapidxml::xml_node<>* val = document->allocate_node(rapidxml::node_element, "Value");
				val->value(document->allocate_string(std::to_string(x).c_str()));
				weights->append_node(val);
			}

			for (double x : m_bias_matrix)
			{
				rapidxml::xml_node<>* val = document->allocate_node(rapidxml::node_element, "Value");
				val->value(document->allocate_string(std::to_string(x).c_str()));
				bias->append_node(val);
			}

			layer_node->append_node(weights);
			layer_node->append_node(bias);
		}


		VectorXd NeuronDensePart::getNodeValues(rapidxml::xml_node<>* node, const int& size)
		{
			VectorXd out = VectorXd::Zero(size);
			rapidxml::xml_node<>* val = node->first_node("Value");
			for (int i = 0; i < size; ++i)
			{
				out(i) = std::stod(val->value());
				val = val->next_sibling("Value");
			}
			return out;
		}


		MatrixXd NeuronDensePart::getMatrixFromVector(const int& rows, const int& cols, const VectorXd& data)
		{
			MatrixXd out(rows, cols);
			for (int i = 0; i < rows * cols; ++i)
				out(i) = data(i);
			return out;
		}
	}
}
