#pragma once
#include "src/activations/Activations.h"
#include "src/optimizers/Optimizers.h"
#include "src/CSV/CSVParser.h"
#include <algorithm>


namespace nn
{
	enum Activations { TANH, RELU, SIGMOID, SOFTMAX };

	/*
	struct LayerData
	{
		MatrixXd weights;
		VectorXd bias;
	};
	*/

	struct Data
	{
		VectorXd target;
		VectorXd values;
	};

	struct DataSet
	{
		std::vector<Data> data;
	};

	class NeuralNetwork
	{
	private:
		std::vector<Layer*> m_layers;
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
		//NeuralNetwork(std::string filename);
		~NeuralNetwork();

		VectorXd predict(VectorXd input_vals);
		VectorXd predict(std::vector<double> input_vals);
		double meanSquaredError(VectorXd true_output, VectorXd predicted_output);
		VectorXd meanSquaredErrorPrime(VectorXd true_output, VectorXd predicted_output);
		void train(CSVParser& parser, int epochs, const Optimizer& optimizer);
		//bool saveNetworkToFile(std::string filename);

	private:
		std::vector<unsigned> getNodeIntValues(rapidxml::xml_node<>* node);
		VectorXd getNodeValues(rapidxml::xml_node<>* node, int num_of_values);
		VectorXd vectorToEigenMatrix(const std::vector<double>& input_vector);
		VectorXd labelToEigenMatrix(int label);
		VectorXd vectorToEigenVector(const std::vector<double>& input_vector);
		MatrixXd eigenVectorToEigenMatrix(const VectorXd& input_vector, int rows, int cols);
		void fillDataSet(CSVParser& parser, int& num_of_samples);
		void updateDataValues(const std::vector<double>& new_values, const double& new_target, double& min_value, double& max_value, std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& targets);
		void normalizeValues(std::vector<std::vector<double>>& values, const double& min_value, const double& max_value);
		void normalizeVector(std::vector<double>& vec, double min_value, double max_value);
		void fillLayers();

		std::vector<double> createVectorFromLabel(double label);
		/*
		template <class T, class Z>
		void fillLayers(const std::vector<LayerData>& layers_data);
		template <class T, class Z>
		void setActivationClasses(const std::vector<LayerData>& layers_data);
		*/

		Layer* createActivationLayer(int type);
	};
}
