#pragma once
#ifndef _NEURAL_NET_
#define _NEURAL_NET_
#include"Neuron.h"
#include"Layer.h"

class NeuralNet
{
private:
	int numInputs;
	int numOutputs;
	int numHiddenLayers;
	int numNeuronsPerLayer;
	//double responsTreshold; //ako bude trebaloza softmax

	std::vector<Layer*>* layers;
	double* outputLayer;

public:
	NeuralNet(int inputs,
			  int outputs,
			  int hiddenLayers,
			  int neuronsPerLayer);
	~NeuralNet();
	double* getWeights() const;
	std::vector<double> softMaxPrime(std::vector<double> input, int n);
	void feedForward(std::vector<double>* inputs,
					 std::vector<double>* outputLayer,
					 const double bias);
	double calcDeltas();
	void backPropagation(vector<double>* outputs, int teachFact);
	//funkcije aktivacije Relu i SoftMAx
	inline double  relu(double inputValue);
	std::vector<double> softMax(std::vector<double> inputInSoftMaxFromOutpuLayer);

};


#endif // !_NEURAL_NET_