#pragma once
#include "../addons/EigenRand/EigenRand"
#include "../addons/rapidxml/rapidxml.hpp"
#include "../addons/rapidxml/rapidxml_print.hpp"
#include <iostream>


using namespace Eigen;


class Layer
{
protected:
	VectorXd m_input_matrix;

public:
	virtual VectorXd feedForward(VectorXd input_vals) = 0;
	virtual VectorXd backPropagation(VectorXd gradient, double learning_rate) = 0;
	virtual bool isNeuronDensePart() const { return false; }
};
