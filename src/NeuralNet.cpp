#include "NeuralNet.h"
#include<math.h>


NeuralNet::NeuralNet(int inputs,
					 int outputs,
					 int hiddenLayers,
					 int neuronsPerLayer,
					 int learningReat, int numEpoch)
{
	numInputs = inputs;
	numOutputs = outputs;
	numHiddenLayers = hiddenLayers;
	numNeuronsPerLayer = neuronsPerLayer;
	this->numEpocha = numEpoch;
	this->learningRate = learningReat;
	layers = new vector<Layer*>(hiddenLayers + 2);

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
				sum += n->getWeight(k) * upstream->getNeurons(k)->getValue() + bias;
			}
			//Ovde ce mozda trebadati staviti uslov kada je i = 2, 
			//tada radi sofaMax za  izlazne neurone
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


inline std::vector<double> NeuralNet::softMax(std::vector<double> inputInSoftMaxFromOutpuLayer) {
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
inline std::vector<double> NeuralNet::softMaxPrime(std::vector<double> input, int n) {
	std::vector<double> derivateOut;
	std::vector<double>::iterator begin = input.begin();
	std::vector<double>::iterator end = input.end();
	int it = 0;

	std::vector<double> o1 = softMax(input);
	std::vector<double> o2;
	while (begin != end)
	{
		o2[it] = 1.0 - o1[it];
		derivateOut[it] = o2[it] * o1[it];
		it++;
	}
	o2.clear();
	o1.clear();

	return derivateOut;
}

void NeuralNet::backPropagation(vector<double>* outputs, int teachFact) {
	Layer *outputLayer = (*layers)[numHiddenLayers + 1];
	for (int i = 0; i < outputs->size(); i++) {
		Neuron *n = outputLayer->getNeurons(i);
		double adjusted = -n->getValue();
		if (i == teachFact) {
			adjusted += 1;
		}
		n->setDelta(relu(n->getActivation()) * adjusted);
	}

	//U slucaju da budemo trebali raditi sa softMax derivate
	//std::vector<double> actvOutLay;
	//std::vector<double> sofMPrime;;
	//for (int i = 0; i < outputLayer->getNumNeur();i++) {
	//	actvOutLay[i] = outputLayer->getNeurons(i)->getActivation();
	//}
	//sofMPrime = softMaxPrime(actvOutLay,1);
	//actvOutLay.clear();

	//	for (int i = 0; i < outputs->size(); i++) {
	//	Neuron *n = outputLayer->getNeurons(i);
	//	double adjusted = -n->getValue();
	//	if (i == teachFact) {
	//		adjusted += 1;
	//	}
	//	n->setDelta(sofMPrime[i] * adjusted);
	//}

	for (int i = numHiddenLayers; i >= 0; i--) {
		Layer *currHidden = (*layers)[i];
		Layer *downOutLayer = (*layers)[i + 1];

		for (int j = 0; j < currHidden->getNumNeur(); j++) {
			double sum = 0;
			Neuron *n = currHidden->getNeurons(j);
			for (int k = 0; k < downOutLayer->getNumNeur();k++) {
				sum += learningRate * downOutLayer->getNeurons(j)->getWeight(i) *
					downOutLayer->getNeurons(j)->getDelta();
			}

			n->setDelta(relu(n->getActivation()) * sum);

			for (int l = 0; l < downOutLayer->getNumNeur(); l++) {
				downOutLayer->getNeurons(l)->updateWeightsOnPos(i, learningRate * 
				relu(n->getActivation()) * downOutLayer->getNeurons(l)->getDelta());
			}
		}
	}




}

void NeuralNet::training(int MAX_ITERATION, std::vector<std::vector<double>*> *inputsSetOfimages, double targets[]) {

	double sumError = 0;
	int numEp = 0;
	double minErr = 0.01;
	do
	{
		int bias = 1;
		std::vector<double>* outputLayers = new vector<double>(10);
		
		for (int i = 0; i < inputsSetOfimages->size(); i++) {
			
			feedForward((*inputsSetOfimages)[i],outputLayers,bias);

			for (int j = 0; j < (*layers)[numHiddenLayers + 1]->getNumNeur();j++) {
				sumError += pow(targets[j]-(*layers)[numHiddenLayers+1]->getNeurons(j)->getOutput(),2);
			}
			backPropagation(outputLayers, learningRate);
		}
		sumError /= 2;
		numEp++;
		std::cerr << "Ukupna greska = " << sumError << "\t" << "i broj epoha = " << numEp << std::flush;
	} while (numEp<numEpocha && sumError < minErr);

}