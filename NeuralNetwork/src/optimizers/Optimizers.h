#pragma once
#include "../../assets/EigenRand/EigenRand"
#include <iostream>


using namespace Eigen;

namespace nn
{
	namespace optimizer
	{
		class Optimizer
		{
		protected:
			double m_learning_rate;

		public:
			Optimizer(const double& learning_rate) : m_learning_rate(learning_rate) {}
			Optimizer(const Optimizer& other, const int& weights_matrix_rows, const int& weights_matrix_cols) : m_learning_rate(other.m_learning_rate) {}
			virtual Matrix<double, Dynamic, Dynamic> getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient) = 0;
			virtual VectorXd getDeltaBias(const VectorXd& gradient) = 0;
			virtual Optimizer* clone(const int& weights_matrix_rows, const int& weights_matrix_cols) const = 0;
		};


		class SGD : public Optimizer
		{
		private:
			double m_momentum;
			MatrixXd m_momentum_weights;
			VectorXd m_momentum_bias;

		public:
			SGD(const double& learning_rate = 0.001, const double& momentum = 0.9);
			SGD(const SGD& other, const int& weights_matrix_rows, const int& weights_matrix_cols);
			Matrix<double, Dynamic, Dynamic> getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient);
			VectorXd getDeltaBias(const VectorXd& gradient);
			SGD* clone(const int& weights_matrix_rows, const int& weights_matrix_cols) const override { return new SGD(*this, weights_matrix_rows, weights_matrix_cols); }
		};


		class AdaGrad : public Optimizer
		{
		private:
			double m_epsilon;
			double m_initial_accumulator;
			MatrixXd m_accumulated_weights;
			VectorXd m_accumulated_bias;

		public:
			AdaGrad(const double& learning_rate = 0.001, const double& initial_accumulator = 0.1, const double& epsilon = 1e-8);
			AdaGrad(const AdaGrad& other, const int& weights_matrix_rows, const int& weights_matrix_cols);
			Matrix<double, Dynamic, Dynamic> getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient);
			VectorXd getDeltaBias(const VectorXd& gradient);
			AdaGrad* clone(const int& weights_matrix_rows, const int& weights_matrix_cols) const override { return new AdaGrad(*this, weights_matrix_rows, weights_matrix_cols); }
		};


		class RMSprop : public Optimizer
		{
		private:
			double m_epsilon;
			double m_decay_factor;
			MatrixXd m_accumulated_weights;
			VectorXd m_accumulated_bias;

		public:
			RMSprop(const double& learning_rate = 0.001, const double& decay_factor = 0.9, const double& epsilon = 1e-8);
			RMSprop(const RMSprop& other, const int& weights_matrix_rows, const int& weights_matrix_cols);
			Matrix<double, Dynamic, Dynamic> getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient);
			VectorXd getDeltaBias(const VectorXd& gradient);
			RMSprop* clone(const int& weights_matrix_rows, const int& weights_matrix_cols) const override { return new RMSprop(*this, weights_matrix_rows, weights_matrix_cols); }
		};

		class Adam : public Optimizer
		{
		private:
			unsigned m_t_weights = 1;
			unsigned m_t_bias = 1;
			double m_epsilon;
			double m_beta1;
			double m_beta2;
			MatrixXd m_weights_first_momentum;
			MatrixXd m_weights_second_momentum;
			VectorXd m_bias_first_momentum;
			VectorXd m_bias_second_momentum;

		public:
			Adam(const double& learning_rate = 0.001, const double& beta1 = 0.9, const double& beta2 = 0.999, const double& epsilon = 1e-8);
			Adam(const Adam& other, const int& weights_matrix_rows, const int& weights_matrix_cols);
			Matrix<double, Dynamic, Dynamic> getDeltaWeights(const VectorXd& input_vector, const VectorXd& gradient);
			VectorXd getDeltaBias(const VectorXd& gradient);
			Adam* clone(const int& weights_matrix_rows, const int& weights_matrix_cols) const override { return new Adam(*this, weights_matrix_rows, weights_matrix_cols); }
		};
	}
}
