#include "../NuralNetwork.h"
#include <chrono>
#include <iomanip>
#include <algorithm>


namespace nn
{
	NeuralNetwork::NeuralNetwork(
		const std::vector<unsigned>& topology, 
		const int& activation_function, 
		const int& output_activation_function, 
		const int& hidden_layer_initializer, 
		const int& output_layer_initializer
	) :
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
		fillLayers(
			m_layers, 
			m_topology, 
			m_activation_function, 
			m_output_activation_function, 
			hidden_layer_initializer, 
			output_layer_initializer
		);
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
		m_layers(layers),
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
		case loss::BINARY_CROSS_ENTROPY:
			return new loss::BinaryCrossEntropy();
		case loss::QUADRATIC:
			return new loss::Quadratic();
		default:
			return nullptr;
		}
	}


	void NeuralNetwork::fillLayers
	(
		std::vector<layer::Layer*>& layers, 
		std::vector<unsigned>& topology, 
		const int& activation_function, 
		const int& output_activation_function, 
		const int& hidden_layer_initializer, 
		const int& output_layer_initializer, 
		rapidxml::xml_node<>* layers_node
	)
	{
		rapidxml::xml_node<>* layer_node = nullptr;
		if (layers_node)
			layer_node = layers_node->first_node("Layer");
		for (auto iter = topology.begin(); iter < topology.end() - 1; ++iter)
		{
			if (layer_node)
			{
				layers.push_back(new layer::NeuronDensePart(*iter, *(iter + 1), layer_node));
				layer_node = layer_node->next_sibling("Layer");
			}

			if (std::next(iter) == topology.end() - 1)
			{
				if (!layer_node)
					layers.push_back(new layer::NeuronDensePart(*iter, *(iter + 1), output_layer_initializer));
				layers.push_back(createActivationLayer(output_activation_function));
			}
			else
			{
				if (!layer_node)
					layers.push_back(new layer::NeuronDensePart(*iter, *(iter + 1), hidden_layer_initializer));
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


	void NeuralNetwork::shuffleDataSet(std::vector<Data>& dataset, std::default_random_engine& random_engine)
	{
		random_engine.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
		std::shuffle(dataset.begin(), dataset.end(), random_engine);
	}


	void NeuralNetwork::trainOnBatch(const std::vector<Data>& batch, double& error, loss::Loss* loss_func, const optimizer::Optimizer& optimizer)
	{
		for (auto iter = batch.begin(); iter < batch.end(); ++iter)
		{
			VectorXd output = predict((*iter).values);
			VectorXd gradient = loss_func->calcLossPrime((*iter).target, output);
			error += loss_func->calcLoss((*iter).target, output);

			for (auto iter = m_layers.rbegin(); iter != m_layers.rend(); ++iter)
			{
				gradient = (*iter)->backPropagation(gradient, optimizer);
			}
		}
	}


	void NeuralNetwork::showDebuggingData
	(
		const int& epoch, 
		const int& max_epoch, 
		const double& duration_s, 
		const double& duration_ms, 
		const double& error, 
		const double& accuracy
	)
	{
		std::cout << epoch << "/" << max_epoch << " - ";
		std::cout << std::format("{:.2}", duration_s) << "s - ";
		std::cout << std::format("{:.2}", duration_ms) << "ms/step - ";
		std::cout << "loss: " << std::format("{:.10}", error);
		if (accuracy > 0)
		{
			std::cout << " - accuracy: " << std::format("{:.10}", accuracy);
		}
		std::cout << std::endl;
	}


	double NeuralNetwork::evaluate(std::vector<Data>& dataset)
	{
		int correctPredictions = 0;
		
		for (const auto& sample : dataset) 
		{
			Eigen::VectorXd predicted = predict(sample.values);

			if (predicted.size() > 1)
			{
				Eigen::Index predicted_class_index;
				predicted.maxCoeff(&predicted_class_index);

				Eigen::Index actual_class_index;
				sample.target.maxCoeff(&actual_class_index);

				if (predicted_class_index == actual_class_index)
					correctPredictions++;
			}
			else
			{
				int predictedClass = (predicted(0) > 0.5) ? 1 : 0;
				if (predictedClass == sample.target(0))
					correctPredictions++;
			}
		}

		return static_cast<double>(correctPredictions) / dataset.size();
	}


	void NeuralNetwork::train
	(
		std::vector<Data>& dataset, 
		const int& batch_size, 
		const double& validation_split, 
		const int& epochs, 
		const optimizer::Optimizer& optimizer, 
		const int& loss_function
	)
	{
		auto rng = std::default_random_engine{};
		shuffleDataSet(dataset, rng);
		loss::Loss* loss_func = createLossFunction(loss_function);
		size_t training_size = dataset.size() - static_cast<size_t>(validation_split * dataset.size());
		std::vector<Data> training_data(dataset.begin(), dataset.begin() + training_size);
		std::vector<Data> test_data(dataset.begin() + training_size, dataset.end());

		for (int i = 0; i < epochs; i++)
		{
			shuffleDataSet(training_data, rng);
			double error = 0.0;
			double accuracy = 0.0;
			auto start = std::chrono::high_resolution_clock::now();

			for (unsigned batch_num = 0; batch_num < training_data.size(); batch_num += batch_size)
			{
				std::vector<Data> batch(
					training_data.begin() + batch_num, 
					std::min(training_data.begin() + batch_num + batch_size, 
					training_data.end())
				);
				trainOnBatch(batch, error, loss_func, optimizer);
			}

			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			error /= dataset.size();

			if (validation_split > 0)
				accuracy = evaluate(test_data);

			showDebuggingData(
				i + 1, 
				epochs, 
				((double)duration.count() / 1000), 
				((double)duration.count() / (double)dataset.size()),
				error, 
				accuracy
			);
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
		std::unique_ptr<rapidxml::xml_document<>> document(new rapidxml::xml_document<>());
		rapidxml::xml_node<>* root = document->allocate_node(rapidxml::node_element, "NeuralNetwork");
		rapidxml::xml_node<>* topology = document->allocate_node(rapidxml::node_element, "Topology");
		rapidxml::xml_node<>* activation_function = document->allocate_node(rapidxml::node_element, "ActivationFunction");
		rapidxml::xml_node<>* out_activation_function = document->allocate_node(rapidxml::node_element, "OutputActivationFunction");
		rapidxml::xml_node<>* layers = document->allocate_node(rapidxml::node_element, "Layers");

		for (auto iter = m_topology.begin(); iter < m_topology.end(); ++iter)
		{
			rapidxml::xml_node<>* val = document->allocate_node(rapidxml::node_element, "Value");
			val->value(document->allocate_string(std::to_string(*iter).c_str()));
			topology->append_node(val);
		}

		activation_function->value(document->allocate_string(std::to_string(m_activation_function).c_str()));
		out_activation_function->value(document->allocate_string(std::to_string(m_output_activation_function).c_str()));

		for (layer::Layer* _layer : m_layers)
		{
			if (_layer->getType() == layer::DENSE)
			{
				layer::NeuronDensePart* dense = dynamic_cast<layer::NeuronDensePart*>(_layer);
				rapidxml::xml_node<>* layer_node = document->allocate_node(rapidxml::node_element, "Layer");
				dense->saveLayer(document.get(), layer_node);
				layers->append_node(layer_node);
			}
		}

		root->append_node(topology);
		root->append_node(activation_function);
		root->append_node(out_activation_function);
		root->append_node(layers);
		document->append_node(root);

		std::ofstream file(filename);
		file << *document;
		file.close();
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
		std::unique_ptr<rapidxml::xml_document<>> document(new rapidxml::xml_document<>());
		document->parse<0>(&buffer[0]);

		rapidxml::xml_node<>* root = document->first_node("NeuralNetwork");
		rapidxml::xml_node<>* topology = root->first_node("Topology");
		rapidxml::xml_node<>* activation_function = topology->next_sibling("ActivationFunction");
		rapidxml::xml_node<>* out_activation_function = activation_function->next_sibling("OutputActivationFunction");
		rapidxml::xml_node<>* layers = out_activation_function->next_sibling("Layers");

		_topology = getNodeUnsignedValues(topology);
		activation = std::stoi(activation_function->value());
		out_activation = std::stoi(out_activation_function->value());
		fillLayers(
			_layers, 
			_topology, 
			activation,
			out_activation, 
			0, 
			0, 
			layers
		);

		file.close();

		return NeuralNetwork(_topology, activation, out_activation, data_min, data_max, _layers);
	}
}
