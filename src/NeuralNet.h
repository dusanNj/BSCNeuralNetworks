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
	int numEpocha;
	double learningRate;
	double responseThreshold;
	//double responsTreshold; //ako bude trebaloza softmax

	std::vector<Layer*>* layers;
	double* outputLayer;

public:
	NeuralNet(int inputs,
			  int outputs,
			  int hiddenLayers,
			  int neuronsPerLayer,
		double learningReat, int numEpoch,double responseThreshold);
	~NeuralNet();
	double* getWeights() const;
	inline std::vector<double> softMaxPrime(std::vector<double> input, int n);
	void feedForward(std::vector<double>* inputs,
					 std::vector<double>* outputLayer,
					 const double bias);
	//double calcDeltas();
	void backPropagation(vector<double>* outputs, int teachFact);
	
	void training(int MAX_ITERATION,  std::vector<std::vector<double>*> *inputsSetOfimages, std::vector<double> targets);
	void feedForward2(std::vector<double>* inputs,std::vector<double>* outputLayer,const double bias);
	
	//funkcije aktivacije Relu i SoftMAx
	inline double  relu(double inputValue);
	inline std::vector<double> softMax(std::vector<double> inputInSoftMaxFromOutpuLayer);

	inline double NeuralNet::sigmoid(double activation) {
		return 1.0 / (1.0 + exp(-activation / responseThreshold));
	}

	// Compute the derivative of the sigmoid function
	inline double NeuralNet::sigmoidPrime(double activation) {
		double exponential = exp(activation / responseThreshold);
		return exponential / (responseThreshold * pow(exponential + 1, 2));
	}

	double calculateError(std::vector<double> targets);

	void updateCorectionWeights();

};


#endif // !_NEURAL_NET_