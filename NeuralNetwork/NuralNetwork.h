#pragma once
#include "src/activations/Activations.h"
#include "src/optimizers/Optimizers.h"
#include "src/CSV/CSVParser.h"
#include "src/losses/Losses.h"
#include <algorithm>


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
		int m_input_size;
		int m_activation_function;
		int m_output_activation_function;

	public:
		NeuralNetwork(std::vector<unsigned>& topology, int activation_function, int output_activation_function);
		NeuralNetwork(const std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function, const double& data_min, const double& data_max, const std::vector<layer::Layer*>& layers);
		~NeuralNetwork();

		VectorXd predict(VectorXd input_vals);
		VectorXd predict(std::vector<double> input_vals);
		void train(CSVParser& parser, const int& epochs, const optimizer::Optimizer& optimizer, const int& loss_function);
		void saveModel(const char* filename);
		static NeuralNetwork loadModel(const char* filename);

	private:
		VectorXd vectorToEigenMatrix(const std::vector<double>& input_vector);
		VectorXd labelToEigenMatrix(int label);
		VectorXd vectorToEigenVector(const std::vector<double>& input_vector);
		MatrixXd eigenVectorToEigenMatrix(const VectorXd& input_vector, int rows, int cols);
		void fillDataSet(CSVParser& parser, int& num_of_samples);
		void updateDataValues(const std::vector<double>& new_values, const double& new_target, double& min_value, double& max_value, std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& targets);
		void normalizeValues(std::vector<std::vector<double>>& values, const double& min_value, const double& max_value);
		void normalizeVector(std::vector<double>& vec, double min_value, double max_value);
		static void fillLayers(std::vector<layer::Layer*>& layers, std::vector<unsigned>& topology, const int& activation_function, const int& output_activation_function, rapidxml::xml_node<>* layers_node = nullptr);

		std::vector<double> createVectorFromLabel(double label);
		static layer::Layer* createActivationLayer(int type);
		loss::Loss* createLossFunction(int type);
		static std::vector<unsigned> getNodeUnsignedValues(rapidxml::xml_node<>* node);
	};
}
