#pragma once
#include "../addons/EigenRand/EigenRand"
#include "../addons/rapidxml/rapidxml.hpp"
#include "../addons/rapidxml/rapidxml_print.hpp"
#include <iostream>


using namespace Eigen;

namespace nn
{
	class Optimizer
	{
	protected:
		double m_learning_rate;

	public:
		Optimizer(double learning_rate) : m_learning_rate(learning_rate) {}
		Optimizer(const Optimizer& other, int weights_matrix_rows, int weights_matrix_cols) : m_learning_rate(other.m_learning_rate) {}
		virtual Matrix<double, Dynamic, Dynamic> getDeltaWeights(VectorXd input_vector, VectorXd gradient) = 0;
		virtual VectorXd getDeltaBias(VectorXd gradient) = 0;
		virtual Optimizer* clone(int weights_matrix_rows, int weights_matrix_cols) const = 0;
	};


	class SGD : public Optimizer
	{
	private:
		double m_momentum;
		Matrix<double, Dynamic, Dynamic> m_momentum_weights;
		VectorXd m_momentum_bias;

	public:
		SGD(double learning_rate = 0.01, double momentum = 0.9);
		SGD(const SGD& other, int weights_matrix_rows, int weights_matrix_cols);
		Matrix<double, Dynamic, Dynamic> getDeltaWeights(VectorXd input_vector, VectorXd gradient);
		VectorXd getDeltaBias(VectorXd gradient);
		SGD* clone(int weights_matrix_rows, int weights_matrix_cols) const override { return new SGD(*this, weights_matrix_rows, weights_matrix_cols); }
	};
}