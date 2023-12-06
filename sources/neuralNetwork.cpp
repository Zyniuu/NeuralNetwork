#include "../headers/neuralNetwork.h"


NeuralNetwork::NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction)
{
	_topology = topology;
	_activationFunction = activationFunction;
	_outputActivationFunction = outputActivationFunction;
	if (!_topology.empty())
		labels = _topology.back();
	switch (activationFunction)
	{
	case TANH:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<TanhActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<TanhActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<TanhActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<TanhActivation, SoftmaxActivation>();
			break;
		}
		break;
	case RELU:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<ReLUActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<ReLUActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<ReLUActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<ReLUActivation, SoftmaxActivation>();
			break;
		}
		break;
	case SIGMOID:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<SigmoidActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<SigmoidActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<SigmoidActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<SigmoidActivation, SoftmaxActivation>();
			break;
		}
		break;
	case SOFTMAX:
		switch (outputActivationFunction)
		{
		case RELU:
			fillLayers<SoftmaxActivation, ReLUActivation>();
			break;
		case SIGMOID:
			fillLayers<SoftmaxActivation, SigmoidActivation>();
			break;
		case TANH:
			fillLayers<SoftmaxActivation, TanhActivation>();
			break;
		case SOFTMAX:
			fillLayers<SoftmaxActivation, SoftmaxActivation>();
			break;
		}
		break;
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

Matrix<double, Dynamic, 1> NeuralNetwork::labelToEigenMatrix(int label)
{
	Matrix<double, Dynamic, 1> result = MatrixXd::Zero(labels, 1);
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

Matrix<double, Dynamic, 1> NeuralNetwork::predict(Matrix<double, Dynamic, 1> inputVals)
{
	Matrix<double, Dynamic, 1> output = normalizeVector(inputVals);
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

Matrix<double, Dynamic, 1> NeuralNetwork::predict(std::vector<double> inputVals)
{
	Matrix<double, Dynamic, 1> output = vectorToEigenMatrix(inputVals);
	output = normalizeVector(output);
	for (Layer* layer : layers)
	{
		output = layer->feedForward(output);
	}
	return output;
}

Matrix<double, Dynamic, 1> NeuralNetwork::vectorToEigenMatrix(const std::vector<double>& inputVector)
{
	Map<const VectorXd> eigenMap(inputVector.data(), inputVector.size());
	VectorXd eigenMatrix = eigenMap;
	return eigenMatrix;
}

Matrix<double, Dynamic, 1> NeuralNetwork::normalizeVector(Matrix<double, Dynamic, 1>& vectorToNormalize)
{
	minInputValue = vectorToNormalize.minCoeff() < minInputValue ? vectorToNormalize.minCoeff() : minInputValue;
	maxInputValue = vectorToNormalize.maxCoeff() > maxInputValue ? vectorToNormalize.maxCoeff() : maxInputValue;
	if (minInputValue == maxInputValue)
		return vectorToNormalize;
	return (vectorToNormalize.array() - minInputValue) / (maxInputValue - minInputValue);
}

double NeuralNetwork::meanSquaredError(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output)
{
	Matrix<double, Dynamic, 1> _error = (true_output - predicted_output).array().square();
	return _error.mean();
}

Matrix<double, Dynamic, 1> NeuralNetwork::meanSquaredErrorPrime(Matrix<double, Dynamic, 1> true_output, Matrix<double, Dynamic, 1> predicted_output)
{
	double scale = 2.0 / true_output.size();
	return scale * (predicted_output - true_output);
}

void NeuralNetwork::train(CSVParser& parser, int epochs, double learning_rate)
{
	int numberOfSamples = parser.countLines();
	parser.getDataFromSingleLine();
	Matrix<double, Dynamic, 1> temp = vectorToEigenMatrix(parser.getValues());
	maxInputValue = temp.maxCoeff();
	minInputValue = temp.minCoeff();
	parser.restartFile();
	for (int i = 0; i < epochs; i++)
	{
		double _error = 0.0;
		while (!parser.endOfFile())
		{
			parser.getDataFromSingleLine();
			Matrix<double, Dynamic, 1> inputVals = vectorToEigenMatrix(parser.getValues());
			Matrix<double, Dynamic, 1> output = predict(inputVals);
			Matrix<double, Dynamic, 1> y;
			if (_topology.back() > 1)
				y = labelToEigenMatrix(parser.getTarget());
			else
			{
				y.resize(1);
				y(0) = parser.getTarget();
			}
			_error += meanSquaredError(y, output);
			Matrix<double, Dynamic, 1> gradient = meanSquaredErrorPrime(y, output);
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
	std::cout << "Starting the save function..." << std::endl;
	// Create document root 
	std::cout << "Creating tree root..." << std::endl;
	rapidxml::xml_document<> doc;
	rapidxml::xml_node<>* root = doc.allocate_node(rapidxml::node_element, "NeuralNetwork");
	doc.append_node(root);
	// Create topology node
	std::cout << "Saving topology structure..." << std::endl;
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
	std::cout << "Saving activation nodes structure..." << std::endl;
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
	std::cout << "Creating node for layers..." << std::endl;
	rapidxml::xml_node<>* layersNode = doc.allocate_node(rapidxml::node_element, "Layers");
	root->append_node(layersNode);

	for (Layer* layer : layers)
	{
		if (layer->isNeuronDensePart())
		{
			std::cout << "Saving layer weights..." << std::endl;
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
			std::cout << "Saving layer bias..." << std::endl;
			Matrix<double, Dynamic, 1> biasMatrix = densePart->getBiasMatrix();
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
	std::cout << "Saving to file..." << std::endl;
	std::ofstream file(filename);
	file << doc;
	file.close();
	std::cout << "Freeing memory..." << std::endl;
	doc.clear();
	std::cout << "DONE" << std::endl;
	return true;
}