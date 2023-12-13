#pragma once
#include "src/activations/Activations.h"
#include "src/optimizers/Optimizers.h"
#include "src/CSV/CSVParser.h"
#include "src/losses/Losses.h"


namespace nn
{
	struct Data
	{
		VectorXd target;
		VectorXd values;
	};

	class NeuralNetwork
	{
	private:
		std::vector<layer::Layer*> m_layers;
		std::vector<unsigned> m_topology;
		std::vector<Data> m_dataset;
		double m_min_value;
		double m_max_value;
		int m_output_size;
		int m_activation_function;
		int m_output_activation_function;

	public:
		NeuralNetwork(const std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function);
		NeuralNetwork(const std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function, const double& data_min, const double& data_max, const std::vector<layer::Layer*>& layers);
		~NeuralNetwork();

		VectorXd predict(const VectorXd& input_vals);
		VectorXd predict(std::vector<double> input_vals);
		void train(CSVParser& parser, const int& batch_size, const int& epochs, const optimizer::Optimizer& optimizer, const int& loss_function);
		void saveModel(const char* filename);

		static NeuralNetwork loadModel(const char* filename);

	private:
		VectorXd vectorToEigenMatrix(const std::vector<double>& input_vector);
		VectorXd vectorToEigenVector(const std::vector<double>& input_vector);
		void fillDataSet(CSVParser& parser, int& num_of_samples);
		void updateDataValues(const std::vector<double>& new_values, const double& new_target, double& min_value, double& max_value, std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& targets);
		void normalizeValues(std::vector<std::vector<double>>& values, const double& min_value, const double& max_value);
		void normalizeVector(std::vector<double>& vec, const double& min_value, const double& max_value);
		void trainOnBatch(const std::vector<Data>& batch, double& error, loss::Loss* loss_func, const optimizer::Optimizer& optimizer);
		std::vector<double> createVectorFromLabel(const unsigned& label);
		void shuffleDataSet(std::vector<Data>& dataset, std::default_random_engine& random_engine);
		loss::Loss* createLossFunction(const int& type);

		static std::vector<unsigned> getNodeUnsignedValues(rapidxml::xml_node<>* node);
		static layer::Layer* createActivationLayer(const int& type);
		static void fillLayers(std::vector<layer::Layer*>& layers, std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function, rapidxml::xml_node<>* layers_node = nullptr);
	};
}
