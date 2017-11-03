#pragma once
#ifndef _NEURON_
#define _NEURON_
#include<iostream>
#include<stdio.h>
#include<vector>
#include<random>
#include<chrono>
using namespace std;


class Neuron
{
private:
	double bias;
	vector<double>* weights;
	double output;
	double value;
	double delta;
	double activation;
	int numOfInputs;

public:
	Neuron(int inputs);
	~Neuron();

	//setMet
	void setBias(double bias) {
		this->bias = bias;
	}
	void setOutput(double output) {
		this->output = output;
	}
	void setValue(double value) {
		this->value = value;
	}
	void setDelta(double delta) {
		this->delta = delta;
	}
	void setInputs(int numOfinputs) {
		this->numOfInputs = numOfinputs;
	}

	double getWeight(int n) const {
		return (*weights)[n];
	}

	void updateWeightsOnPos(int pos, double val) {
		(*weights)[pos] = val;
	}

	void setActivation(int a) {
		activation = a;
	}
	//getMetod
	double getActivation() {
		return activation;
	}
	double getBias() {
		return this->bias;
	}
	double getOutput() {
		return this->output;
	}
	double getValue() {
		return this->value;
	}

	double getDelta() {
		return this->delta;
	}
	double getNumInputs() {
		return this->numOfInputs;
	}
	double  printWeights() {
		for (int i = 0; i < weights->size(); i++) {
			cout << (*weights)[i] << " ";
		}
		cout << "\n";
	}

	//-----------------------------
	void setRandWeights();
	void setRandBias();
};


#endif // !_NEURON_


