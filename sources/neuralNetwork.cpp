#include "../headers/neuralNetwork.h"


NeuralNetwork::NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction)
	: _topology(topology), _activationFunction(activationFunction), _outputActivationFunction(outputActivationFunction)
{
	if (!_topology.empty())
		labels = _topology.back();
	ActivationPair activationPair
	{ 
		static_cast<ActivationFunction>(_activationFunction),
		static_cast<ActivationFunction>(_outputActivationFunction) 
	};
	if (activationMap.find(activationPair) != activationMap.end())
	{
		activationMap[activationPair](std::vector<LayerData>());
	}
	else
	{
		std::cerr << "Error: Activation pair not found in map." << std::endl;
		exit(3);
	}
}

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

	std::vector<LayerData> layersData;
	rapidxml::xml_node<>* node = root->first_node();

	while (node)
	{
		std::string nodeName = node->name();
		if (nodeName == "Topology")
		{
			_topology = getNodeIntValues(node);
		}
		else if (nodeName == "ActivationFunction")
			_activationFunction = std::stoi(node->value());
		else if (nodeName == "OutputActivationFunction")
			_outputActivationFunction = std::stoi(node->value());
		else if (nodeName == "Layers")
		{
			rapidxml::xml_node<>* layerNode = node->first_node("Layer");
			std::vector<unsigned>::iterator iter = _topology.begin();
			for (iter; iter < _topology.end() - 1; iter++)
			{
				int inputSize = *iter;
				int outputSize = *(iter + 1);
				rapidxml::xml_node<>* weightsNode = layerNode->first_node("Weights");
				VectorXd tempVec = getNodeValues(weightsNode, inputSize * outputSize);
				MatrixXd weights = eigenVectorToEigenMatrix(tempVec, outputSize, inputSize);
				rapidxml::xml_node<>* biasNode = layerNode->first_node("Bias");
				VectorXd bias = getNodeValues(biasNode, outputSize);
				layersData.push_back({ weights, bias });
				layerNode = layerNode->next_sibling("Layer");
			}
		}
		node = node->next_sibling();
	}

	file.close();

	ActivationPair activationPair
	{
		static_cast<ActivationFunction>(_activationFunction),
		static_cast<ActivationFunction>(_outputActivationFunction)
	};
	if (activationMap.find(activationPair) != activationMap.end())
	{
		activationMap[activationPair](layersData);
	}
	else
	{
		std::cerr << "Error: Activation pair not found in map." << std::endl;
		exit(3);
	}
}

NeuralNetwork::~NeuralNetwork()
{
	for (Layer* layer : layers) 
	{
		delete layer;
	}
	layers.clear();
}

std::vector<unsigned> NeuralNetwork::getNodeIntValues(rapidxml::xml_node<>* node)
{
	std::vector<unsigned> out;
	rapidxml::xml_node<>* valueNode = node->first_node("Value");
	while (valueNode)
	{
		out.push_back(std::stoi(valueNode->value()));
		valueNode = valueNode->next_sibling("Value");
	}
	return out;
}

VectorXd NeuralNetwork::getNodeValues(rapidxml::xml_node<>* node, int numOfValues)
{
	VectorXd out;
	out.resize(numOfValues);
	int i = 0;
	rapidxml::xml_node<>* valueNode = node->first_node("Value");
	while (valueNode)
	{
		out(i) = std::stod(valueNode->value());
		valueNode = valueNode->next_sibling("Value");
		i++;
	}
	return out;
}

template <class T, class Z>
void NeuralNetwork::setActivationClasses(const std::vector<LayerData>& layersData)
{
	if (layersData.size() <= 0)
		fillLayers<T, Z>();
	else
		fillLayers<T, Z>(layersData);
}

VectorXd NeuralNetwork::labelToEigenMatrix(int label)
{
	VectorXd result = MatrixXd::Zero(labels, 1);
	if (label >= 0 && label < labels)
		result(label, 0) = 1.0;
	return result;
}

template <class T, class Z>
void NeuralNetwork::fillLayers()
{
	std::vector<unsigned>::iterator iter = _topology.begin();
	for (iter; iter < _topology.end() - 1; iter++)
	{
		layers.push_back(new NeuronDensePart(*iter, *(iter + 1)));
		if (std::next(iter) == _topology.end() - 1)
			layers.push_back(new Z());
		else
			layers.push_back(new T());
	}
}

template <class T, class Z>
void NeuralNetwork::fillLayers(const std::vector<LayerData>& layersData)
{
	auto iter = layersData.begin();
	for (iter; iter < layersData.end(); iter++)
	{
		layers.push_back(new NeuronDensePart(iter->weights, iter->bias));
		if (std::next(iter) == layersData.end())
			layers.push_back(new Z());
		else
			layers.push_back(new T());
	}
}

VectorXd NeuralNetwork::predict(VectorXd inputVals)
{
	VectorXd output = normalizeVector(inputVals);
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

VectorXd NeuralNetwork::predict(std::vector<double> inputVals)
{
	VectorXd output = vectorToEigenMatrix(inputVals);
	output = normalizeVector(output);
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

MatrixXd NeuralNetwork::eigenVectorToEigenMatrix(const VectorXd& inputVector, int rows, int cols)
{
	MatrixXd temp(cols, rows);
	for (int i = 0; i < cols * rows; i++)
		temp(i) = inputVector(i);
	MatrixXd out(rows, cols);
	out = temp.transpose();
	return out;
}

VectorXd NeuralNetwork::vectorToEigenMatrix(const std::vector<double>& inputVector)
{
	Map<const VectorXd> eigenMap(inputVector.data(), inputVector.size());
	VectorXd eigenMatrix = eigenMap;
	return eigenMatrix;
}

VectorXd NeuralNetwork::normalizeVector(VectorXd& vectorToNormalize)
{
	minInputValue = vectorToNormalize.minCoeff() < minInputValue ? vectorToNormalize.minCoeff() : minInputValue;
	maxInputValue = vectorToNormalize.maxCoeff() > maxInputValue ? vectorToNormalize.maxCoeff() : maxInputValue;
	if (minInputValue == maxInputValue)
		return vectorToNormalize;
	return (vectorToNormalize.array() - minInputValue) / (maxInputValue - minInputValue);
}

double NeuralNetwork::meanSquaredError(VectorXd true_output, VectorXd predicted_output)
{
	VectorXd _error = (true_output - predicted_output).array().square();
	return _error.mean();
}

VectorXd NeuralNetwork::meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output)
{
	double scale = 2.0 / true_output.size();
	return scale * (predicted_output - true_output);
}

void NeuralNetwork::train(CSVParser& parser, int epochs, double learning_rate)
{
	int numberOfSamples = parser.countLines();
	parser.getDataFromSingleLine();
	VectorXd temp = vectorToEigenMatrix(parser.getValues());
	maxInputValue = temp.maxCoeff();
	minInputValue = temp.minCoeff();
	parser.restartFile();
	for (int i = 0; i < epochs; i++)
	{
		double _error = 0.0;
		while (!parser.endOfFile())
		{
			parser.getDataFromSingleLine();
			VectorXd inputVals = vectorToEigenMatrix(parser.getValues());
			VectorXd output = predict(inputVals);
			VectorXd y;
			if (_topology.back() > 1)
				y = labelToEigenMatrix(parser.getTarget());
			else
			{
				y.resize(1);
				y(0) = parser.getTarget();
			}
			_error += meanSquaredError(y, output);
			VectorXd gradient = meanSquaredErrorPrime(y, output);
			for (auto iter = layers.rbegin(); iter != layers.rend(); ++iter)
			{
				gradient = (*iter)->backPropagation(gradient, learning_rate);
			}
		}
		_error /= numberOfSamples;
		std::cout << i + 1 << "/" << epochs << "\terror = " << _error << std::endl;
		parser.restartFile();
	}
}

bool NeuralNetwork::saveNetworkToFile(std::string filename)
{
	std::cout << "Saving the neural network..." << std::endl;
	// Create document root 
	rapidxml::xml_document<> doc;
	rapidxml::xml_node<>* root = doc.allocate_node(rapidxml::node_element, "NeuralNetwork");
	doc.append_node(root);
	// Create topology node
	rapidxml::xml_node<>* topologyNode = doc.allocate_node(rapidxml::node_element, "Topology");
	root->append_node(topologyNode);
	// Attach values to topology
	std::vector<unsigned>::iterator iter = _topology.begin();
	for (iter; iter < _topology.end(); iter++)
	{
		rapidxml::xml_node<>* valueNode = doc.allocate_node(rapidxml::node_element, "Value");
		valueNode->value(doc.allocate_string(std::to_string(*iter).c_str()));
		topologyNode->append_node(valueNode);
	}
	// Create nodes for activation functions inside root node
	root->append_node(
		doc.allocate_node(
			rapidxml::node_element, 
			"ActivationFunction", 
			doc.allocate_string(std::to_string(_activationFunction).c_str())
		)
	);
	root->append_node(
		doc.allocate_node(
			rapidxml::node_element, 
			"OutputActivationFunction", 
			doc.allocate_string(std::to_string(_outputActivationFunction).c_str())
		)
	);
	// Crete node for layers inside root node
	rapidxml::xml_node<>* layersNode = doc.allocate_node(rapidxml::node_element, "Layers");
	root->append_node(layersNode);

	for (Layer* layer : layers)
	{
		if (layer->isNeuronDensePart())
		{
			// Create node for each dense layer
			rapidxml::xml_node<>* layerNode = doc.allocate_node(rapidxml::node_element, "Layer");
			layersNode->append_node(layerNode);
			NeuronDensePart* densePart = dynamic_cast<NeuronDensePart*>(layer);
			// Add weights node to the layer
			rapidxml::xml_node<>* weightsNode = doc.allocate_node(rapidxml::node_element, "Weights");
			layerNode->append_node(weightsNode);
			// Add bias node to the layer
			rapidxml::xml_node<>* biasNode = doc.allocate_node(rapidxml::node_element, "Bias");
			layerNode->append_node(biasNode);
			// To the weights node add individual weights from the weights matrix
			Matrix<double, Dynamic, Dynamic> weightsMatrix = densePart->getWeightsMatrix();
			for (int i = 0; i < weightsMatrix.rows(); i++)
			{
				for (int j = 0; j < weightsMatrix.cols(); j++)
				{
					double value = weightsMatrix(i, j);
					weightsNode->append_node(
						doc.allocate_node(
							rapidxml::node_element,
							"Value",
							doc.allocate_string(std::to_string(value).c_str())
						)
					);
				}
			}
			// To the bias node add weights from the bias weights matrix
			VectorXd biasMatrix = densePart->getBiasMatrix();
			for (int i = 0; i < biasMatrix.size(); i++)
			{
				double value = biasMatrix(i);
				biasNode->append_node(
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