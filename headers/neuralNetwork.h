#pragma once
#include "neuronDensePart.h"
#include "tanhActivation.h"
#include "ReLUActivation.h"
#include "sigmoidActivation.h"
#include "softmaxActivation.h"
#include "CSVParser.h"


enum ActivationFunction { TANH, RELU, SIGMOID, SOFTMAX };

struct ActivationPair 
{
	ActivationFunction activation1;
	ActivationFunction activation2;

	bool operator==(const ActivationPair& other) const 
	{
		return std::tie(activation1, activation2) == std::tie(other.activation1, other.activation2);
	}
};

namespace std 
{
	template <>
	struct hash<ActivationPair> 
	{
		std::size_t operator()(const ActivationPair& activationPair) const 
		{
			auto hash1 = std::hash<int>()(static_cast<int>(activationPair.activation1));
			auto hash2 = std::hash<int>()(static_cast<int>(activationPair.activation2));
			return hash1 ^ hash2;
		}
	};
}

struct LayerData
{
	MatrixXd weights;
	VectorXd bias;
};

class NeuralNetwork
{
private:
	std::vector<Layer*> layers;
	std::vector<unsigned> _topology;
	int labels;
	int _activationFunction;
	int _outputActivationFunction;
	double maxInputValue;
	double minInputValue;

	std::unordered_map<ActivationPair, std::function<void(const std::vector<LayerData>&)>> activationMap
	{
		{{ TANH,    TANH    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<TanhActivation,    TanhActivation>(layersData);    }},
		{{ TANH,    RELU    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<TanhActivation,    ReLUActivation>(layersData);    }},
		{{ TANH,    SIGMOID }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<TanhActivation,    SigmoidActivation>(layersData); }},
		{{ TANH,    SOFTMAX }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<TanhActivation,    SoftmaxActivation>(layersData); }},
		{{ RELU,    TANH    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<ReLUActivation,    TanhActivation>(layersData);    }},
		{{ RELU,    RELU    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<ReLUActivation,    ReLUActivation>(layersData);    }},
		{{ RELU,    SIGMOID }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<ReLUActivation,    SigmoidActivation>(layersData); }},
		{{ RELU,    SOFTMAX }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<ReLUActivation,    SoftmaxActivation>(layersData); }},
		{{ SIGMOID, TANH    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SigmoidActivation, TanhActivation>(layersData);    }},
		{{ SIGMOID, RELU    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SigmoidActivation, ReLUActivation>(layersData);    }},
		{{ SIGMOID, SIGMOID }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SigmoidActivation, SigmoidActivation>(layersData); }},
		{{ SIGMOID, SOFTMAX }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SigmoidActivation, SoftmaxActivation>(layersData); }},
		{{ SOFTMAX, TANH    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SoftmaxActivation, TanhActivation>(layersData);    }},
		{{ SOFTMAX, RELU    }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SoftmaxActivation, ReLUActivation>(layersData);    }},
		{{ SOFTMAX, SIGMOID }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SoftmaxActivation, SigmoidActivation>(layersData); }},
		{{ SOFTMAX, SOFTMAX }, [this](const std::vector<LayerData>& layersData) { this->setActivationClasses<SoftmaxActivation, SoftmaxActivation>(layersData); }},
	};

	std::vector<unsigned> getNodeIntValues(rapidxml::xml_node<>* node);
	VectorXd getNodeValues(rapidxml::xml_node<>* node, int numOfValues);
	VectorXd vectorToEigenMatrix(const std::vector<double>& inputVector);
	VectorXd labelToEigenMatrix(int label);
	VectorXd normalizeVector(VectorXd& vectorToNormalize);
	MatrixXd eigenVectorToEigenMatrix(const VectorXd& inputVector, int rows, int cols);
	template <class T, class Z>
	void fillLayers();
	template <class T, class Z>
	void fillLayers(const std::vector<LayerData>& layersData);
	template <class T, class Z>
	void setActivationClasses(const std::vector<LayerData>& layersData);
public:
	NeuralNetwork(std::vector<unsigned>& topology, int activationFunction, int outputActivationFunction);
	NeuralNetwork(std::string filename);
	~NeuralNetwork();
	VectorXd predict(VectorXd inputVals);
	VectorXd predict(std::vector<double> inputVals);
	double meanSquaredError(VectorXd true_output, VectorXd predicted_output);
	VectorXd meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output);
	void train(CSVParser& parser, int epochs, double learning_rate);
	bool saveNetworkToFile(std::string filename);
};
