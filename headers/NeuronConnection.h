#pragma once
#include <cstdlib>

class NeuronConnection
{
private:
	double weight;
	double deltaWeight;
public:
	NeuronConnection();
	double getWeight() const;
	void setWeight(const double value);
	double getDeltaWeight() const;
	void setDeltaWeight(const double value);
};
