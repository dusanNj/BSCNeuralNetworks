#include "NeuralNet.h"
#include<math.h>


NeuralNet::NeuralNet(int inputs,
					 int outputs,
					 int hiddenLayers,
					 int neuronsPerLayer)
{
	numInputs = inputs;
	numOutputs = outputs;
	numHiddenLayers = hiddenLayers;
	numNeuronsPerLayer = neuronsPerLayer;

	//Inicijalizacija layera
	(*layers)[0] = new Layer(inputs, 0);
	(*layers)[1] = new Layer(neuronsPerLayer, inputs);
	(*layers)[hiddenLayers + 1] = new Layer(outputs, neuronsPerLayer);
	//Ispitati zasto ovo radi u for petlji
	for (int i = 2; i < layers->size() - 1; i++) {
		(*layers)[i] = new Layer(neuronsPerLayer, neuronsPerLayer);
	}
}


NeuralNet::~NeuralNet()
{
	for (int i = 0; i < layers->size(); i++) {
		delete (*layers)[i];
	}
	delete layers;
}

inline double NeuralNet::relu(double inputValue) {
	if (inputValue < 0) {
		return 0;
	}
	else
	{
		return inputValue;
	}
}

void NeuralNet::feedForward(std::vector<double>* inputs,
							std::vector<double>* outputLayer,
							const double bias)
{
	//Za prvi layer setujemo vrednosti neurona sto predstavljaju vrednosti piksela
	Layer* inputLayer = (*layers)[0];
	for (int i = 0; i < inputLayer->getNumNeur(); i++) {
		inputLayer->getNeurons(i)->setValue((*inputs)[i]);
	}

	for (int i = 1; i < numHiddenLayers + 2 ; i++) {
		Layer *curr = (*layers)[i];
		Layer *upstream = (*layers)[i - 1];

		for (int j = 0; j < curr->getNumNeur(); j++) {
			Neuron *n = curr->getNeurons(j);
			double sum = 0;
			for (int k = 0; k < upstream->getNumNeur(); k++) {
				sum += n->getWeight(k) * upstream->getNeurons(k)->getValue();
			}
			n->setActivation(sum);
			n->setValue(relu(sum));
		}
	}

	Layer* lastLayer = (*layers)[numHiddenLayers + 1];
	for (int i = 0; i, lastLayer->getNumNeur(); i++) {
		(*outputLayer)[i] = lastLayer->getNeurons(i)->getValue();
	}
	(*outputLayer) = softMax(*outputLayer);
}


std::vector<double> NeuralNet::softMax(std::vector<double> inputInSoftMaxFromOutpuLayer) {
	double sumOut = 0;
	for (int j = 0; j < inputInSoftMaxFromOutpuLayer.size(); j++)
	{
		sumOut += exp(inputInSoftMaxFromOutpuLayer[j]);
	}
	double output = 0;
	std::vector<double> allOutput;
	allOutput.resize(inputInSoftMaxFromOutpuLayer.size());
	for (int i=0; i< inputInSoftMaxFromOutpuLayer.size(); i++)
	{
		output = exp(inputInSoftMaxFromOutpuLayer[i]) / sumOut;
		allOutput[i] = output;
	}
	return allOutput;
}
std::vector<double> NeuralNet::softMaxPrime(std::vector<double> input, int n) {

}

//void NeuralNet::backPropagation(vector<double>* outputs, int teachFact) {
//	Layer *outputLayer = (*layers)[numHiddenLayers + 1];
//
//	for (int i = 0; i < outputs->size(); i++) {
//		Neuron *n = outputLayer->getNeurons(i);
//		double adjust = n->getValue();
//
//		if (i == teachFact) {
//			adjust += 1;
//		}
//		n->setDelta();
//	}
//}

double NeuralNet::calcDeltas() {
	double error = 0;
	//greska u izlaznom sloju
	Layer *outputLayer = (*layers)[numHiddenLayers + 1];
	for (int i = 0; i < outputLayer->getNumNeur(); i++) {

	}
}
