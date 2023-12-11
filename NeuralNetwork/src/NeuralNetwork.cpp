#include "../NuralNetwork.h"


namespace nn
{
	NeuralNetwork::NeuralNetwork(std::vector<unsigned>& topology, int activation_function, int output_activation_function) : 
		m_topology(topology),
		m_activation_function(activation_function),
		m_output_activation_function(output_activation_function),
		m_min_value(),
		m_max_value(),
		m_input_size(topology.front()),
		m_output_size(topology.back()),
		m_dataset()
	{
		if (m_topology.empty())
		{
			std::cerr << "ERROR: enterd topology is incorrect." << std::endl;
			exit(EXIT_FAILURE);
		}
		fillLayers();
	}


	NeuralNetwork::~NeuralNetwork()
	{
		for (Layer* layer : m_layers)
		{
			delete layer;
		}
		m_layers.clear();
		m_dataset.clear();
	}


	Layer* NeuralNetwork::createActivationLayer(int type)
	{
		switch (type)
		{
		case TANH:
			return new TanhActivation();
		case RELU:
			return new ReLUActivation();
		case SIGMOID:
			return new SigmoidActivation();
		case SOFTMAX:
			return new SoftmaxActivation();
		default:
			return nullptr;
		}
	}


	std::vector<unsigned> NeuralNetwork::getNodeIntValues(rapidxml::xml_node<>* node)
	{
		std::vector<unsigned> out;
		rapidxml::xml_node<>* value_node = node->first_node("Value");
		while (value_node)
		{
			out.push_back(std::stoi(value_node->value()));
			value_node = value_node->next_sibling("Value");
		}
		return out;
	}


	VectorXd NeuralNetwork::getNodeValues(rapidxml::xml_node<>* node, int num_of_values)
	{
		VectorXd out;
		out.resize(num_of_values);
		int i = 0;
		rapidxml::xml_node<>* value_node = node->first_node("Value");
		while (value_node)
		{
			out(i) = std::stod(value_node->value());
			value_node = value_node->next_sibling("Value");
			i++;
		}
		return out;
	}


	VectorXd NeuralNetwork::labelToEigenMatrix(int label)
	{
		VectorXd result = MatrixXd::Zero(m_output_size, 1);
		if (label >= 0 && label < m_output_size)
			result(label, 0) = 1.0;
		return result;
	}


	void NeuralNetwork::fillLayers()
	{
		std::vector<unsigned>::iterator iter = m_topology.begin();
		for (iter; iter < m_topology.end() - 1; ++iter)
		{
			m_layers.push_back(new NeuronDensePart(*iter, *(iter + 1)));
			if (std::next(iter) == m_topology.end() - 1)
				m_layers.push_back(createActivationLayer(m_output_activation_function));
			else
				m_layers.push_back(createActivationLayer(m_activation_function));
		}
	}


	VectorXd NeuralNetwork::predict(VectorXd input_vals)
	{
		VectorXd output = input_vals;
		for (Layer* layer : m_layers)
		{
			output = layer->feedForward(output);
		}
		return output;
	}


	VectorXd NeuralNetwork::predict(std::vector<double> input_vals)
	{
		normalizeVector(input_vals, m_min_value, m_max_value);
		VectorXd output = vectorToEigenMatrix(input_vals);
		for (Layer* layer : m_layers)
		{
			output = layer->feedForward(output);
		}
		return output;
	}


	MatrixXd NeuralNetwork::eigenVectorToEigenMatrix(const VectorXd& input_vector, int rows, int cols)
	{
		MatrixXd temp(cols, rows);
		for (int i = 0; i < cols * rows; i++)
			temp(i) = input_vector(i);
		MatrixXd out(rows, cols);
		out = temp.transpose();
		return out;
	}


	VectorXd NeuralNetwork::vectorToEigenMatrix(const std::vector<double>& input_vector)
	{
		Map<const VectorXd> eigen_map(input_vector.data(), input_vector.size());
		VectorXd eigen_matrix = eigen_map;
		return eigen_matrix;
	}


	double NeuralNetwork::meanSquaredError(VectorXd true_output, VectorXd predicted_output)
	{
		VectorXd error = (true_output - predicted_output).array().square();
		return error.mean();
	}


	VectorXd NeuralNetwork::meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output)
	{
		double scale = 2.0 / true_output.size();
		return scale * (predicted_output - true_output);
	}


	std::vector<double> NeuralNetwork::createVectorFromLabel(double label)
	{
		std::vector<double> label_vector(m_output_size);
		std::fill(label_vector.begin(), label_vector.end(), 0);
		label_vector[m_output_size > 1 ? label : 0] = m_output_size > 1 ? 1 : label;
		return label_vector;
	}


	void NeuralNetwork::updateDataValues(const std::vector<double>& new_values, const double& new_target, double& min_value, double& max_value, std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& targets)
	{
		values.push_back(new_values);
		targets.push_back(createVectorFromLabel(new_target));
		double temp_min = *std::min_element(values.back().begin(), values.back().end());
		double temp_max = *std::max_element(values.back().begin(), values.back().end());

		min_value = (temp_min < min_value) ? temp_min : min_value;
		max_value = (temp_max > max_value) ? temp_max : max_value;
	}


	void NeuralNetwork::normalizeValues(std::vector<std::vector<double>>& values, const double& min_value, const double& max_value)
	{
		if ((min_value == max_value) || (min_value == 0 && max_value == 1))
			return;

		for (auto iter = values.begin(); iter < values.end(); ++iter)
		{
			normalizeVector(*iter, min_value, max_value);
		}
	}


	void NeuralNetwork::normalizeVector(std::vector<double>& vec, double min_value, double max_value)
	{
		std::for_each(vec.begin(), vec.end(),
			[min_value, max_value](double& x)
			{
				x = (x - min_value) / (max_value - min_value);
			}
		);
	}


	VectorXd NeuralNetwork::vectorToEigenVector(const std::vector<double>& input_vector)
	{
		return Map<const VectorXd>(input_vector.data(), input_vector.size());
	}


	void NeuralNetwork::fillDataSet(CSVParser& parser, int& num_of_samples)
	{
		double min_value = std::numeric_limits<double>::max(), max_value = -std::numeric_limits<double>::max();
		std::vector<std::vector<double>> values, targets;

		while (!parser.endOfFile())
		{
			parser.getDataFromSingleLine();
			num_of_samples += 1;
			updateDataValues(parser.getValues(), parser.getTarget(), min_value, max_value, values, targets);
		}

		parser.~CSVParser();

		normalizeValues(values, min_value, max_value);

		for (unsigned i = 0; i < values.size(); ++i)
		{
			Data temp;
			temp.values = vectorToEigenVector(values[i]);
			temp.target = vectorToEigenVector(targets[i]);
			m_dataset.push_back(temp);
		}

		m_min_value = min_value;
		m_max_value = max_value;
	}


	void NeuralNetwork::train(CSVParser& parser, int epochs, const Optimizer& optimizer)
	{
		int number_of_samples = 0;
		fillDataSet(parser, number_of_samples);
		for (int i = 0; i < epochs; i++)
		{
			double error = 0.0;
			for (auto iter = m_dataset.begin(); iter < m_dataset.end(); ++iter)
			{
				VectorXd output = predict((*iter).values);
				VectorXd gradient = meanSquaredErrorPrime((*iter).target, output);
				error += meanSquaredError((*iter).target, output);

				for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter)
				{
					gradient = (*iter)->backPropagation(gradient, optimizer);
				}
			}
			error /= number_of_samples;
			std::cout << i + 1 << "/" << epochs << "\terror = " << error << std::endl;
			parser.restartFile();
		}
	}


	/*
	template <class T, class Z>
	void NeuralNetwork::setActivationClasses(const std::vector<LayerData>& layers_data)
	{
		if (layers_data.size() <= 0)
			fillLayers<T, Z>();
		else
			fillLayers<T, Z>(layers_data);
	}
	*/


	/*
	template <class T, class Z>
	void NeuralNetwork::fillLayers(const std::vector<LayerData>& layers_data)
	{
		auto iter = layers_data.begin();
		for (iter; iter < layers_data.end(); iter++)
		{
			m_layers.push_back(new NeuronDensePart(iter->weights, iter->bias));
			if (std::next(iter) == layers_data.end())
				m_layers.push_back(new Z());
			else
				m_layers.push_back(new T());
		}
	}
	*/


	/*
	bool NeuralNetwork::saveNetworkToFile(std::string filename)
	{
		std::cout << "Saving the neural network..." << std::endl;
		// Create document root 
		rapidxml::xml_document<> doc;
		rapidxml::xml_node<>* root = doc.allocate_node(rapidxml::node_element, "NeuralNetwork");
		doc.append_node(root);
		// Create topology node
		rapidxml::xml_node<>* topology_node = doc.allocate_node(rapidxml::node_element, "Topology");
		root->append_node(topology_node);
		// Attach values to topology
		std::vector<unsigned>::iterator iter = m_topology.begin();
		for (iter; iter < m_topology.end(); iter++)
		{
			rapidxml::xml_node<>* value_node = doc.allocate_node(rapidxml::node_element, "Value");
			value_node->value(doc.allocate_string(std::to_string(*iter).c_str()));
			topology_node->append_node(value_node);
		}
		// Create nodes for activation functions inside root node
		root->append_node(
			doc.allocate_node(
				rapidxml::node_element,
				"ActivationFunction",
				doc.allocate_string(std::to_string(m_activation_function).c_str())
			)
		);
		root->append_node(
			doc.allocate_node(
				rapidxml::node_element,
				"OutputActivationFunction",
				doc.allocate_string(std::to_string(m_output_activation_function).c_str())
			)
		);
		// Crete node for layers inside root node
		rapidxml::xml_node<>* layers_node = doc.allocate_node(rapidxml::node_element, "Layers");
		root->append_node(layers_node);

		for (Layer* layer : m_layers)
		{
			if (layer->isNeuronDensePart())
			{
				// Create node for each dense layer
				rapidxml::xml_node<>* layer_node = doc.allocate_node(rapidxml::node_element, "Layer");
				layers_node->append_node(layer_node);
				NeuronDensePart* dense_part = dynamic_cast<NeuronDensePart*>(layer);
				// Add weights node to the layer
				rapidxml::xml_node<>* weights_node = doc.allocate_node(rapidxml::node_element, "Weights");
				layer_node->append_node(weights_node);
				// Add bias node to the layer
				rapidxml::xml_node<>* bias_node = doc.allocate_node(rapidxml::node_element, "Bias");
				layer_node->append_node(bias_node);
				// To the weights node add individual weights from the weights matrix
				Matrix<double, Dynamic, Dynamic> weights_matrix = dense_part->getWeightsMatrix();
				for (int i = 0; i < weights_matrix.rows(); i++)
				{
					for (int j = 0; j < weights_matrix.cols(); j++)
					{
						double value = weights_matrix(i, j);
						weights_node->append_node(
							doc.allocate_node(
								rapidxml::node_element,
								"Value",
								doc.allocate_string(std::to_string(value).c_str())
							)
						);
					}
				}
				// To the bias node add weights from the bias weights matrix
				VectorXd bias_matrix = dense_part->getBiasMatrix();
				for (int i = 0; i < bias_matrix.size(); i++)
				{
					double value = bias_matrix(i);
					bias_node->append_node(
						doc.allocate_node(
							rapidxml::node_element,
							"Value",
							doc.allocate_string(std::to_string(value).c_str())
						)
					);
				}
			}
		}
		// Save tree to file and free memory
		std::ofstream file(filename);
		file << doc;
		file.close();
		std::cout << "Freeing memory..." << std::endl;
		doc.clear();
		std::cout << "DONE" << std::endl;
		return true;
	}
	*/


	/*
	NeuralNetwork::NeuralNetwork(std::string filename)
	{
		std::ifstream file(filename);
		if (!file.is_open())
		{
			std::cerr << "Can't open file " << filename << std::endl;
			exit(3);
		}
		std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		buffer.push_back('\0');
		rapidxml::xml_document<> doc;
		doc.parse<0>(&buffer[0]);

		rapidxml::xml_node<>* root = doc.first_node("NeuralNetwork");
		if (root == nullptr)
		{
			std::cerr << "Incorrect file format" << std::endl;
			exit(3);
		}

		std::vector<LayerData> layers_data;
		rapidxml::xml_node<>* node = root->first_node();

		while (node)
		{
			std::string node_name = node->name();
			if (node_name == "Topology")
			{
				m_topology = getNodeIntValues(node);
			}
			else if (node_name == "ActivationFunction")
				m_activation_function = std::stoi(node->value());
			else if (node_name == "OutputActivationFunction")
				m_output_activation_function = std::stoi(node->value());
			else if (node_name == "Layers")
			{
				rapidxml::xml_node<>* layer_node = node->first_node("Layer");
				std::vector<unsigned>::iterator iter = m_topology.begin();
				for (iter; iter < m_topology.end() - 1; iter++)
				{
					int input_size = *iter;
					int output_size = *(iter + 1);
					rapidxml::xml_node<>* weights_node = layer_node->first_node("Weights");
					VectorXd temp_vec = getNodeValues(weights_node, input_size * output_size);
					MatrixXd weights = eigenVectorToEigenMatrix(temp_vec, output_size, input_size);
					rapidxml::xml_node<>* bias_node = layer_node->first_node("Bias");
					VectorXd bias = getNodeValues(bias_node, output_size);
					layers_data.push_back({ weights, bias });
					layer_node = layer_node->next_sibling("Layer");
				}
			}
			node = node->next_sibling();
		}

		file.close();

		ActivationPair activation_pair
		{
			static_cast<ActivationFunction>(m_activation_function),
			static_cast<ActivationFunction>(m_output_activation_function)
		};
		if (m_activation_map.find(activation_pair) != m_activation_map.end())
		{
			m_activation_map[activation_pair](layers_data);
		}
		else
		{
			std::cerr << "Error: Activation pair not found in map." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	*/
}
