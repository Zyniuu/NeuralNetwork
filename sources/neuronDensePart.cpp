#include "../headers/NeuronDensePart.h"


NeuronDensePart::NeuronDensePart(int input_size, int output_size)
{
	// Create matrices for weights and bias with random values in range [-1,1]
	weightsMatrix = MatrixXd::Random(output_size, input_size);
	biasMatrix = MatrixXd::Random(output_size, 1);
	// Data required for Adam optimization
	m_weightsMatrix = MatrixXd::Zero(weightsMatrix.rows(), weightsMatrix.cols());
	v_weightsMatrix = MatrixXd::Zero(weightsMatrix.rows(), weightsMatrix.cols());
	m_biasMatrix = VectorXd::Zero(biasMatrix.size());
	v_biasMatrix = VectorXd::Zero(biasMatrix.size());
}

Matrix<double, Dynamic, 1> NeuronDensePart::feedForward(Matrix<double, Dynamic, 1> inputVals)
{
	inputMatrix = inputVals;
	return weightsMatrix * inputMatrix + biasMatrix;
}

Matrix<double, Dynamic, 1> NeuronDensePart::backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate, int epoch)
{
	//std::cout << m_weightsMatrix << std::endl;
	//std::cout << m_weightsMatrix.rows() << " x " << m_weightsMatrix.cols() <<  std::endl << std::endl;
	//std::cin.get();
	//std::cout << v_weightsMatrix << std::endl;
	//std::cout << v_weightsMatrix.rows() << " x " << v_weightsMatrix.cols() << std::endl << std::endl;
	//std::cin.get();
	// Calculate momentum
	m_weightsMatrix = (beta1 * m_weightsMatrix).colwise() + ((1 - beta1) * gradient);
	//std::cout << m_weightsMatrix << std::endl;
	//std::cout << m_weightsMatrix.rows() << " x " << m_weightsMatrix.cols() << std::endl << std::endl;
	//std::cin.get();
	v_weightsMatrix = (beta2 * v_weightsMatrix).colwise() + ((1 - beta2) * gradient.array().square().matrix());
	//std::cout << v_weightsMatrix << std::endl;
	//std::cout << v_weightsMatrix.rows() << " x " << v_weightsMatrix.cols() << std::endl << std::endl;
	//std::cin.get();
	m_biasMatrix = beta1 * m_biasMatrix + (1 - beta1) * gradient;
	v_biasMatrix = beta2 * v_biasMatrix + (1 - beta2) * gradient.array().square().matrix();
	// Update momentum
	Matrix<double, Dynamic, Dynamic> m_weights_hat = m_weightsMatrix / (1 - std::pow(beta1, epoch + 1));
	//std::cout << m_weights_hat << std::endl;
	//std::cout << m_weights_hat.rows() << " x " << m_weights_hat.cols() << std::endl << std::endl;
	//std::cin.get();
	Matrix<double, Dynamic, Dynamic> v_weights_hat = v_weightsMatrix / (1 - std::pow(beta2, epoch + 1));
	//std::cout << v_weights_hat << std::endl;
	//std::cout << v_weights_hat.rows() << " x " << v_weights_hat.cols() << std::endl << std::endl;
	//std::cin.get();
	Matrix<double, Dynamic, 1> m_bias_hat = m_biasMatrix / (1 - std::pow(beta1, epoch + 1));
	Matrix<double, Dynamic, 1> v_bias_hat = v_biasMatrix / (1 - std::pow(beta2, epoch + 1));
	// apply momentum
	// Matrix<double, Dynamic, Dynamic> gradient_weights = gradient * inputMatrix.transpose();
	Matrix<double, Dynamic, 1> gradient_input = weightsMatrix.transpose() * gradient;
	weightsMatrix.array() -= learning_rate * m_weights_hat.array() / (v_weights_hat.array().sqrt() + eps);
	//std::cout << weightsMatrix << std::endl;
	//std::cout << weightsMatrix.rows() << " x " << weightsMatrix.cols() << std::endl << std::endl;
	//std::cin.get();
	biasMatrix.array() -= learning_rate * m_bias_hat.array() / (v_bias_hat.array().sqrt() + eps);
	return gradient_input;
}