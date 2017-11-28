#include "NeuralNet.h"
#include<math.h>


NeuralNet::NeuralNet(int inputs,
					 int outputs,
					 int hiddenLayers,
					 int neuronsPerLayer,
					 double learningReat, int numEpoch, double responseThreshold)
{
	numInputs = inputs;
	numOutputs = outputs;
	numHiddenLayers = hiddenLayers;
	numNeuronsPerLayer = neuronsPerLayer;
	this->responseThreshold = responseThreshold;
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
		return 1;
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
	//std::clog << "Ulazni layer size: " << inputLayer->getNumNeur() << "\n";
	for (int i = 1; i < numHiddenLayers + 2 ; i++) {
		Layer *curr = (*layers)[i];
		Layer *upstream = (*layers)[i - 1];
		//std::clog << "curr layer size: " << curr->getNumNeur() << "\n";
		//std::clog << "upstream layer size: " << upstream->getNumNeur() << "\n";
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
	//std::clog << "lastLayer layer size: " << lastLayer->getNumNeur() << "\n";
	for (int i = 0; i< lastLayer->getNumNeur(); i++) {
		(*outputLayer)[i] = lastLayer->getNeurons(i)->getValue();
	}
	(*outputLayer) = softMax(*outputLayer);

	for (int i = 0; i < outputLayer->size(); i++) {
		lastLayer->getNeurons(i)->setOutput((*outputLayer)[i]);
	}
}

void NeuralNet::feedForward2(std::vector<double>* inputs,
	std::vector<double>* outputLayer,
	const double bias) {
	//Za prvi layer setujemo vrednosti neurona sto predstavljaju vrednosti piksela
	Layer* inputLayer = (*layers)[0];
	for (int i = 0; i < inputLayer->getNumNeur(); i++) {
		inputLayer->getNeurons(i)->setValue((*inputs)[i]);
	}
	//std::clog << "Ulazni layer size: " << inputLayer->getNumNeur() << "\n";
	for (int i = 1; i < numHiddenLayers + 2; i++) {
		Layer *curr = (*layers)[i];
		Layer *upstream = (*layers)[i - 1];
		//std::clog << "curr layer size: " << curr->getNumNeur() << "\n";
		//std::clog << "upstream layer size: " << upstream->getNumNeur() << "\n";
		for (int j = 0; j < curr->getNumNeur(); j++) {
			Neuron *n = curr->getNeurons(j);
			double sum = 0;
			for (int k = 0; k < upstream->getNumNeur(); k++) {
				sum += n->getWeight(k) * upstream->getNeurons(k)->getValue() /*+ bias*/;
			}
			//Ovde ce mozda trebadati staviti uslov kada je i = 2, 
			//tada radi sofaMax za  izlazne neurone
			n->setActivation(sigmoid(sum));
			n->setValue(sigmoid(sum));
			n->setOutput(sigmoid(sum));
		}
	}

	Layer* lastLayer = (*layers)[numHiddenLayers + 1];
	//std::clog << "lastLayer layer size: " << lastLayer->getNumNeur() << "\n";
	for (int i = 0; i< lastLayer->getNumNeur(); i++) {
		(*outputLayer)[i] = lastLayer->getNeurons(i)->getOutput();
	}
	//(*outputLayer) = softMax(*outputLayer);

	for (int i = 0; i < outputLayer->size(); i++) {
		lastLayer->getNeurons(i)->setOutput((*outputLayer)[i]);
	}
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
		n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
	}

	for (int i = numHiddenLayers; i >= 0; i--) {
		Layer *currHidden = (*layers)[i];
		Layer *downOutLayer = (*layers)[i + 1];

		for (int j = 0; j < currHidden->getNumNeur(); j++) {
			double sum = 0;
			Neuron *n = currHidden->getNeurons(j);
			for (int k = 0; k < downOutLayer->getNumNeur();k++) {
				sum += /*learningRate **/ downOutLayer->getNeurons(k)->getWeight(i) *
					downOutLayer->getNeurons(k)->getDelta();
			}

			n->setDelta(sigmoid(n->getActivation()) * sum);

			for (int l = 0; l < downOutLayer->getNumNeur(); l++) {
				downOutLayer->getNeurons(l)->updateWeightsOnPos(j, learningRate * 
				sigmoid(n->getActivation()) * downOutLayer->getNeurons(l)->getDelta());
			}
		}
	}
}

double NeuralNet::calculateError(std::vector<double> targets) {
	double greska = 0;
	//greska izlaznog sloja
	Layer *layer = (*layers)[numHiddenLayers + 1];
	for (int i = 0; layer->getNumNeur();i++) {
		Neuron *tempN = layer->getNeurons(i);
		double temErr = targets[i] - tempN->getOutput();
		greska += temErr*temErr;
		tempN->setDelta(temErr);
	}

	for (int i = numHiddenLayers; i >= 0; i -- ) {
		Layer *otherLayer = (*layers)[i];

		for (int j = 0; j < otherLayer->getNumNeur();j++) {
			Layer *upLayer = (*layers)[i + 1];
			Neuron *n0 = upLayer->getNeurons(j);
			double sigmaa = 0;
			for (int k = 0; k < upLayer->getNumNeur();k++) {
				Neuron *n1 = upLayer->getNeurons(k);
				sigmaa += n1->getDelta() * n1->getWeight(k);
			}
			double f = n0->getOutput();
			n0->setDelta(f*(1 - f)*sigmaa);
		}
	}

	return greska;
}

//void NeuralNet::updateCorectionWeights() {
//	for (int s = 0; s < numHiddenLayers + 1;s++) {
//		for (int v = 0; v < (*layers)[s + 1]->getNumNeur(); v++) {
//			for (int u = 0; u < (*layers)[s]->getNumNeur();u++) {
//				(*layers)[s + 1]->getNeurons(v)->up(u) = 0.05*(*layers)[s + 1]->getNeurons(v)->getDelta() *
//					(*layers)[s]->getNeurons(u)->getOutput()
//			}
//
//
//
//		}
//	}
//
//}


void NeuralNet::training(int MAX_ITERATION, std::vector<std::vector<double>*> *inputsSetOfimages, std::vector<double> targets) {

	double sumError;
	int numEp = 0;
	double minErr = 0.01;
	std::cout << "broj slika: " << inputsSetOfimages->size() << std::endl;
	do
	{
		int bias = 1;
		std::vector<double>* outputLayers = new vector<double>(10);
		int tempC = inputsSetOfimages->size();
		int nm = (tempC);
		sumError = 0;
		for (int i = 0; i < nm; i++) {
			
			feedForward2((*inputsSetOfimages)[i],outputLayers,bias);

			for (int j = 0; j < (*layers)[numHiddenLayers + 1]->getNumNeur();j++) {
				sumError += pow(targets[j]-(*layers)[numHiddenLayers+1]->getNeurons(j)->getValue(),2);
			}
			backPropagation(outputLayers, learningRate);
			//std::clog <<"\r"<< "Ukupna greska = " << (float)sumError << std::fflush;
		}
		sumError /= 2;
		numEp++;
		std::cerr <<"\nUkupna greska = " << sumError << "\t" << "i broj epoha = " << numEp <<"\n";
	}while (numEp < numEpocha && sumError > minErr);

}