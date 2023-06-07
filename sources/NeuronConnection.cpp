#include "../headers/NeuronConnection.h"

NeuronConnection::NeuronConnection()
{
	weight = rand() / double(RAND_MAX);
	deltaWeight = 0;
}

double NeuronConnection::getWeight() const
{
	return weight;
}

void NeuronConnection::setWeight(const double value)
{
	weight = value;
}

double NeuronConnection::getDeltaWeight() const
{
	return deltaWeight;
}

void NeuronConnection::setDeltaWeight(const double value)
{
	deltaWeight = value;
}
