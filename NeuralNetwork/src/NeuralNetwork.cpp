#include "../NuralNetwork.h"
#include <chrono>
#include <iomanip>


namespace nn
{
	NeuralNetwork::NeuralNetwork(const std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function) :
		m_topology(topology),
		m_activation_function(activation_function),
		m_output_activation_function(output_activation_function),
		m_min_value(),
		m_max_value(),
		m_output_size(topology.back()),
		m_dataset()
	{
		if (m_topology.empty())
		{
			std::cerr << "ERROR: enterd topology is incorrect." << std::endl;
			exit(EXIT_FAILURE);
		}
		fillLayers(m_layers, m_topology, m_activation_function, m_output_activation_function);
	}


	NeuralNetwork::NeuralNetwork
	(
		const std::vector<unsigned>& topology, 
		const int& activation_function, 
		const int& output_activation_function, 
		const double& data_min,
		const double& data_max,
		const std::vector<layer::Layer*>& layers
	) :
		m_topology(topology), 
		m_activation_function(activation_function), 
		m_output_activation_function(output_activation_function), 
		m_min_value(data_min), m_max_value(data_max), 
		m_layers(std::move(layers)),
		m_output_size(topology.back())
	{}


	NeuralNetwork::~NeuralNetwork()
	{
		for (layer::Layer* layer : m_layers)
		{
			delete layer;
		}
		m_layers.clear();
		m_dataset.clear();
	}


	layer::Layer* NeuralNetwork::createActivationLayer(const int& type)
	{
		switch (type)
		{
		case activation::TANH:
			return new activation::TanhActivation();
		case activation::RELU:
			return new activation::ReLUActivation();
		case activation::SIGMOID:
			return new activation::SigmoidActivation();
		case activation::SOFTMAX:
			return new activation::SoftmaxActivation();
		default:
			return nullptr;
		}
	}


	loss::Loss* NeuralNetwork::createLossFunction(const int& type)
	{
		switch (type)
		{
		case loss::MSE:
			return new loss::MeanSquaredError();
		case loss::CROSS_ENTROPY:
			return new loss::CrossEntropy();
		default:
			return nullptr;
		}
	}


	void NeuralNetwork::fillLayers(std::vector<layer::Layer*>& layers, std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function, rapidxml::xml_node<>* layers_node)
	{
		rapidxml::xml_node<>* layer_node = nullptr;
		if (layers_node)
			layer_node = layers_node->first_node("Layer");
		for (auto iter = topology.begin(); iter < topology.end() - 1; ++iter)
		{
			if (!layer_node)
			{
				layers.push_back(new layer::NeuronDensePart(*iter, *(iter + 1)));
			}
			else
			{
				layers.push_back(new layer::NeuronDensePart(*iter, *(iter + 1), layer_node));
				layer_node = layer_node->next_sibling("Layer");
			}

			if (std::next(iter) == topology.end() - 1)
			{
				layers.push_back(createActivationLayer(output_activation_function));
			}
			else
			{
				layers.push_back(createActivationLayer(activation_function));
			}
		}
	}


	VectorXd NeuralNetwork::predict(const VectorXd& input_vals)
	{
		VectorXd output = input_vals;
		for (layer::Layer* layer : m_layers)
		{
			output = layer->feedForward(output);
		}
		return output;
	}


	VectorXd NeuralNetwork::predict(std::vector<double> input_vals)
	{
		normalizeVector(input_vals, m_min_value, m_max_value);
		VectorXd output = vectorToEigenMatrix(input_vals);
		for (layer::Layer* layer : m_layers)
		{
			output = layer->feedForward(output);
		}
		return output;
	}


	VectorXd NeuralNetwork::vectorToEigenMatrix(const std::vector<double>& input_vector)
	{
		Map<const VectorXd> eigen_map(input_vector.data(), input_vector.size());
		VectorXd eigen_matrix = eigen_map;
		return eigen_matrix;
	}


	std::vector<double> NeuralNetwork::createVectorFromLabel(const unsigned& label)
	{
		std::vector<double> label_vector(m_output_size);
		std::fill(label_vector.begin(), label_vector.end(), 0);
		label_vector[m_output_size > 1 ? label : 0] = m_output_size > 1 ? 1 : label;
		return label_vector;
	}


	void NeuralNetwork::updateDataValues(const std::vector<double>& new_values, const double& new_target, double& min_value, double& max_value, std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& targets)
	{
		values.push_back(new_values);
		targets.push_back(createVectorFromLabel((unsigned)new_target));
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


	void NeuralNetwork::normalizeVector(std::vector<double>& vec, const double& min_value, const double& max_value)
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


	void NeuralNetwork::train(CSVParser& parser, const int& epochs, const optimizer::Optimizer& optimizer, const int& loss_function)
	{
		int number_of_samples = 0;
		std::cout << "Filling dataset..." << std::endl;
		fillDataSet(parser, number_of_samples);
		std::cout << "DONE" << std::endl;

		loss::Loss* loss_func = createLossFunction(loss_function);
		for (int i = 0; i < epochs; i++)
		{
			double error = 0.0;
			auto start = std::chrono::high_resolution_clock::now();
			for (auto iter = m_dataset.begin(); iter < m_dataset.end(); ++iter)
			{
				VectorXd output = predict((*iter).values);
				VectorXd gradient = loss_func->calcLossPrime((*iter).target, output);
				error += loss_func->calcLoss((*iter).target, output);

				for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter)
				{
					gradient = (*iter)->backPropagation(gradient, optimizer);
				}
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			error /= number_of_samples;
			std::cout << i + 1 << "/" << epochs << " - ";
			std::cout << std::format("{:.2}", ((float)duration.count() / 1000)) << "s - ";
			std::cout << std::format("{:.2}", ((float)duration.count() / (float)number_of_samples)) << "ms/step - ";
			std::cout << "loss: " << std::format("{:.10}", error) << std::endl;
			parser.restartFile();
		}
		delete loss_func;
	}


	std::vector<unsigned> NeuralNetwork::getNodeUnsignedValues(rapidxml::xml_node<>* node)
	{
		std::vector<unsigned> out;
		rapidxml::xml_node<>* val = node->first_node("Value");
		while (val)
		{
			out.push_back(std::stoi(val->value()));
			val = val->next_sibling("Value");
		}
		return out;
	}


	void NeuralNetwork::saveModel(const char* filename)
	{
		rapidxml::xml_document<> document;
		rapidxml::xml_node<>* root = document.allocate_node(rapidxml::node_element, "NeuralNetwork");
		rapidxml::xml_node<>* topology = document.allocate_node(rapidxml::node_element, "Topology");
		rapidxml::xml_node<>* activation_function = document.allocate_node(rapidxml::node_element, "ActivationFunction");
		rapidxml::xml_node<>* out_activation_function = document.allocate_node(rapidxml::node_element, "OutputActivationFunction");
		rapidxml::xml_node<>* data_min_value = document.allocate_node(rapidxml::node_element, "DataMin");
		rapidxml::xml_node<>* data_max_value = document.allocate_node(rapidxml::node_element, "DataMax");
		rapidxml::xml_node<>* layers = document.allocate_node(rapidxml::node_element, "Layers");

		for (auto iter = m_topology.begin(); iter < m_topology.end(); ++iter)
		{
			rapidxml::xml_node<>* val = document.allocate_node(rapidxml::node_element, "Value");
			val->value(document.allocate_string(std::to_string(*iter).c_str()));
			topology->append_node(val);
		}

		activation_function->value(document.allocate_string(std::to_string(m_activation_function).c_str()));
		out_activation_function->value(document.allocate_string(std::to_string(m_output_activation_function).c_str()));
		data_min_value->value(document.allocate_string(std::to_string(m_min_value).c_str()));
		data_max_value->value(document.allocate_string(std::to_string(m_max_value).c_str()));

		for (layer::Layer* _layer : m_layers)
		{
			if (_layer->getType() == layer::DENSE)
			{
				layer::NeuronDensePart* dense = dynamic_cast<layer::NeuronDensePart*>(_layer);
				rapidxml::xml_node<>* layer_node = document.allocate_node(rapidxml::node_element, "Layer");
				dense->saveLayer(&document, layer_node);
				layers->append_node(layer_node);
			}
		}

		root->append_node(topology);
		root->append_node(activation_function);
		root->append_node(out_activation_function);
		root->append_node(data_min_value);
		root->append_node(data_max_value);
		root->append_node(layers);
		document.append_node(root);

		std::ofstream file(filename);
		file << document;
		file.close();
		document.clear();
	}


	NeuralNetwork NeuralNetwork::loadModel(const char* filename)
	{
		std::vector<layer::Layer*> _layers;
		std::vector<unsigned> _topology;
		int activation, out_activation;
		double data_min, data_max;
		std::ifstream file(filename);

		if (!file.is_open())
		{
			std::cerr << "Can't open file " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		buffer.push_back('\0');
		rapidxml::xml_document<> document;
		document.parse<0>(&buffer[0]);

		rapidxml::xml_node<>* root = document.first_node("NeuralNetwork");
		rapidxml::xml_node<>* topology = root->first_node("Topology");
		rapidxml::xml_node<>* activation_function = topology->next_sibling("ActivationFunction");
		rapidxml::xml_node<>* out_activation_function = activation_function->next_sibling("OutputActivationFunction");
		rapidxml::xml_node<>* data_min_value = out_activation_function->next_sibling("DataMin");
		rapidxml::xml_node<>* data_max_value = data_min_value->next_sibling("DataMax");
		rapidxml::xml_node<>* layers = data_max_value->next_sibling("Layers");

		_topology = getNodeUnsignedValues(topology);
		activation = std::stoi(activation_function->value());
		out_activation = std::stoi(out_activation_function->value());
		data_min = std::stod(data_min_value->value());
		data_max = std::stod(data_max_value->value());
		fillLayers(_layers, _topology, activation, out_activation, layers);

		return NeuralNetwork(_topology, activation, out_activation, data_min, data_max, _layers);
	}
}
