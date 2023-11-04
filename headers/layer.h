#pragma once
#include "../addons/Eigen/Dense"
#include <iostream>


using namespace Eigen;


class Layer
{
public:
	Matrix<double, Dynamic, 1> inputMatrix;
	virtual Matrix<double, Dynamic, 1> feedForward(Matrix<double, Dynamic, 1> inputVals) = 0;
	virtual Matrix<double, Dynamic, 1> backPropagation(Matrix<double, Dynamic, 1> gradient, double learning_rate) = 0;
};
