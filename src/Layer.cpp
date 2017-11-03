#include "Layer.h"



Layer::Layer(int numOfNeuron, int inputsPerNeuron)
{
	setNumNeuron(numOfNeuron);
	creatNeuron(numOfNeuron, inputsPerNeuron);
}


Layer::~Layer()
{
	for (int i = 0; i < neurons->size(); i++) {
		delete (*neurons)[i];
	}
	delete neurons;
}
