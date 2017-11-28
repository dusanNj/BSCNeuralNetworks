#include "Neuron.h"

Neuron::Neuron(int inputs)
{
	setInputs(inputs);
	initializeWeights(inputs);
	setRandWeights();
}


Neuron::~Neuron()
{
	delete weights;
}

void Neuron::setRandWeights() {
	std::vector<double> ws;
	std::mt19937_64 rng;
	// initialize the random number generator with time-dependent seed
	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
	rng.seed(ss);
	// initialize a uniform distribution between 0 and 1
	std::uniform_real_distribution<double> unif(-0.1, 1.0);
	// ready to generate random numbers
	for (int i = 0; i < getNumInputs()+1; i++)
	{
		ws.push_back(unif(rng));
	}
	*weights = ws;
}

void Neuron::setRandBias() {
	std::mt19937_64 rng;
	// initialize the random number generator with time-dependent seed
	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
	rng.seed(ss);
	// initialize a uniform distribution between 0 and 1
	std::uniform_real_distribution<double> unif(-1.0, 1.0);
	// ready to generate random numbers
	double tembias = unif(rng);
	setBias(tembias);
}