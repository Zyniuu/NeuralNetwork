#pragma once
#include "../addons/Eigen/Dense"
#include "../addons/rapidxml/rapidxml.hpp"
#include "../addons/rapidxml/rapidxml_print.hpp"
#include "../addons/rapidxml/rapidxml_iterators.hpp"
#include "../addons/rapidxml/rapidxml_utils.hpp"
#include <iostream>


using namespace Eigen;


class Layer
{
public:
	Matrix<double, Dynamic, 1> inputMatrix;
	virtual Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals) = 0;
	virtual Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate) = 0;
	virtual bool isNeuronDensePart() const { return false; }
};
