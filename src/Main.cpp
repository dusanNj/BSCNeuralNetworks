#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include"NeuralNet.h"
#include "ImageUtility.h"
using namespace std;

using namespace cv;

void getFileFormDirectory(std::string pathToFolder, std::string extension, std::vector<std::string>& returnFileNameList) {

	std::string names;
	std::string search_path = pathToFolder + extension;
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				returnFileNameList.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
}


int main() {

	int inputs = 28 * 28;
	int outPut = 10;
	int hiddenLayers = 1;
	int neuronPerLayer = 20;
	double learRate = 0.05;
	int numE = 100;
	double targeti[] = {0,0,0,0,0,0,0,0,0,1};
	std::vector<double> targets{ 0,0,0,0,0,0,0,0,0,1 };
	std::vector<std::string> imageName;
	double a(50.0), b(28*28);

	std::vector<std::vector<double>*> *InputValuOfPixels = new std::vector<std::vector<double>*>(50);

	std::vector<double> *temp= new std::vector<double>(28*28);

	std::string pathToImages = "I:/DigitsRecognitionBSC/Dataset/9/";

	std::vector<std::vector<double>> ImageValuePixels;
	std::vector<double> tempValue;

	//ImageUtility im;
	getFileFormDirectory(pathToImages,"*.png*",imageName);

	std::vector<string>::iterator begin = imageName.begin();
	std::vector<string>::iterator end = imageName.end();
	int c = 0;
	while (begin != end)
	{
		Mat imgtemp = imread(pathToImages + imageName[c]);
		cvtColor(imgtemp, imgtemp, CV_RGB2GRAY);
		//double a = imgtemp.at<uchar>(1, 1);
		//std::clog << "a: " << a << "\n";
		int temC = 0;
		for (int i = 0; i < imgtemp.cols;i++) {
			for (int j = 0; j < imgtemp.rows;j++) {
				//(*InputValuOfPixels2)->push_back(imgtemp.at<uchar>(i, j));
				tempValue.push_back((imgtemp.at<uchar>(i, j) / 255));
				//(*temp)[temC] = (imgtemp.at<uchar>(i, j)/255);
				temC++;
			}
		}

		ImageValuePixels.push_back(tempValue);
		tempValue.clear();

		begin++;
		c++;
	}

	//InputValuOfPixels = &ImageValuePixels;

	for (int i = 0; i<50;i++) {
		(*InputValuOfPixels)[i] = &ImageValuePixels[i];
	}


	NeuralNet networks(inputs, outPut,hiddenLayers, neuronPerLayer,learRate,numE,1);

	networks.training(100,InputValuOfPixels,targets);


}