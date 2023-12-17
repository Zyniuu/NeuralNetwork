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
		int m_output_size;
		int m_activation_function;
		int m_output_activation_function;

	public:
		NeuralNetwork(
			const std::vector<unsigned>& topology, 
			const int& activation_function, 
			const int& output_activation_function, 
			const int& hidden_layer_initializer, 
			const int& output_layer_initializer
		);
		NeuralNetwork(
			const std::vector<unsigned>& topology, 
			const int& activation_function, 
			const int& output_activation_function,
			const std::vector<layer::Layer*>& layers
		);
		~NeuralNetwork();

		VectorXd predict(const VectorXd& input_vals);
		VectorXd predict(std::vector<double> input_vals);
		void train(
			std::vector<Data>& dataset, 
			const int& batch_size, 
			const double& validation_split, 
			const int& epochs,
			const optimizer::Optimizer& optimizer, 
			const int& loss_function
		);
		double evaluate(std::vector<Data>& dataset);
		void saveModel(const char* filename);

		static NeuralNetwork loadModel(const char* filename);

	private:
		VectorXd vectorToEigenMatrix(const std::vector<double>& input_vector);
		void trainOnBatch(const std::vector<Data>& batch, double& error, loss::Loss* loss_func, const optimizer::Optimizer& optimizer);
		void shuffleDataSet(std::vector<Data>& dataset, std::default_random_engine& random_engine);
		loss::Loss* createLossFunction(const int& type);

		static void showDebuggingData(
			const int& epoch, 
			const int& max_epoch, 
			const double& duration_s, 
			const double& duration_ms, 
			const double& error, 
			const double& accuracy
		);
		static std::vector<unsigned> getNodeUnsignedValues(rapidxml::xml_node<>* node);
		static layer::Layer* createActivationLayer(const int& type);
		static void fillLayers(
			std::vector<layer::Layer*>& layers, 
			std::vector<unsigned>& topology, 
			const int& activation_function, 
			const int& output_activation_function, 
			const int& hidden_layer_initializer, 
			const int& output_layer_initializer, 
			rapidxml::xml_node<>* layers_node = nullptr
		);
	};
}
