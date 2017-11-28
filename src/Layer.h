#pragma once
#ifndef _LAYER_
#define _LAYER_
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include"Neuron.h"

using namespace std;

class Layer
{
private:
	int numNeur;
	vector<Neuron*>* neurons;
public:
	Layer(int numOfNeuron, int inputsPerNeuron);
	~Layer();

	//setMetod
	void setNumNeuron(int numNeur) {
		this->numNeur = numNeur;
	}

	void creatNeuron(int numN, int inpurPN) {
		neurons = new vector<Neuron*>(numN);
		for (int i = 0; i<numN; i++) {
			(*neurons)[i] = new Neuron(inpurPN);
		}
	}

	//getMet
	int getNumNeur() {
		return this->numNeur;
	}
	//neuron na odredjenoj poziciij
	Neuron *getNeurons(int n) const {
		return (*neurons)[n];
	}

	void printNeuron() {
		for (int i = 0; i < neurons->size();i++) {
			cout << "Neuron #" << i << "\n";
			(*neurons)[i]->printWeights();
		}
		cout << "\n\n";
	}
};


#endif // !_LAYER_


