#pragma once
#include "functional.h"


namespace nn
{
	struct LayerData
	{
		MatrixXd weights;
		VectorXd bias;
	};

	class NeuralNetwork
	{
	private:
		std::vector<Layer*> m_layers;
		std::vector<unsigned> m_topology;
		int m_labels;
		int m_activation_function;
		int m_output_activation_function;
		double m_max_input_value;
		double m_min_input_value;

		std::unordered_map<ActivationPair, std::function<void(const std::vector<LayerData>&)>> m_activation_map
		{
			{{ TANH,    TANH    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<TanhActivation,    TanhActivation>(layers_data);    }},
			{{ TANH,    RELU    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<TanhActivation,    ReLUActivation>(layers_data);    }},
			{{ TANH,    SIGMOID }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<TanhActivation,    SigmoidActivation>(layers_data); }},
			{{ TANH,    SOFTMAX }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<TanhActivation,    SoftmaxActivation>(layers_data); }},
			{{ RELU,    TANH    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<ReLUActivation,    TanhActivation>(layers_data);    }},
			{{ RELU,    RELU    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<ReLUActivation,    ReLUActivation>(layers_data);    }},
			{{ RELU,    SIGMOID }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<ReLUActivation,    SigmoidActivation>(layers_data); }},
			{{ RELU,    SOFTMAX }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<ReLUActivation,    SoftmaxActivation>(layers_data); }},
			{{ SIGMOID, TANH    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SigmoidActivation, TanhActivation>(layers_data);    }},
			{{ SIGMOID, RELU    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SigmoidActivation, ReLUActivation>(layers_data);    }},
			{{ SIGMOID, SIGMOID }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SigmoidActivation, SigmoidActivation>(layers_data); }},
			{{ SIGMOID, SOFTMAX }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SigmoidActivation, SoftmaxActivation>(layers_data); }},
			{{ SOFTMAX, TANH    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SoftmaxActivation, TanhActivation>(layers_data);    }},
			{{ SOFTMAX, RELU    }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SoftmaxActivation, ReLUActivation>(layers_data);    }},
			{{ SOFTMAX, SIGMOID }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SoftmaxActivation, SigmoidActivation>(layers_data); }},
			{{ SOFTMAX, SOFTMAX }, [this](const std::vector<LayerData>& layers_data) { this->setActivationClasses<SoftmaxActivation, SoftmaxActivation>(layers_data); }},
		};

		std::vector<unsigned> getNodeIntValues(rapidxml::xml_node<>* node);
		VectorXd getNodeValues(rapidxml::xml_node<>* node, int num_of_values);
		VectorXd vectorToEigenMatrix(const std::vector<double>& input_vector);
		VectorXd labelToEigenMatrix(int label);
		VectorXd normalizeVector(VectorXd& vector_to_normalize);
		MatrixXd eigenVectorToEigenMatrix(const VectorXd& input_vector, int rows, int cols);
		template <class T, class Z>
		void fillLayers();
		template <class T, class Z>
		void fillLayers(const std::vector<LayerData>& layers_data);
		template <class T, class Z>
		void setActivationClasses(const std::vector<LayerData>& layers_data);

	public:
		NeuralNetwork(std::vector<unsigned>& topology, int activation_function, int output_activation_function);
		NeuralNetwork(std::string filename);
		~NeuralNetwork();
		VectorXd predict(VectorXd input_vals);
		VectorXd predict(std::vector<double> input_vals);
		double meanSquaredError(VectorXd true_output, VectorXd predicted_output);
		VectorXd meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output);
		void train(CSVParser& parser, int epochs, const Optimizer& optimizer);
		bool saveNetworkToFile(std::string filename);
	};
}
