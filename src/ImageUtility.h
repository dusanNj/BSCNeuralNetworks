#pragma once
#ifndef _IMAGE_UTILITY_
#define _IMAGE_UTILITY_
#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include"opencv2/opencv.hpp"
#include"opencv2/core/core.hpp"
#include<fstream>
#include<string>
#include<exception>
#include<Windows.h>
using namespace cv;

class ImageUtility
{
private:
	Mat *img;
	int width, height;
	float *pixels;
	bool error;
public:
	static const int NORM_WIDTH = 28;
	static const int NORM_HEIGHT = 28;

	ImageUtility(std::string file, int targetSize, bool boundingBox);
	ImageUtility();
	~ImageUtility();

	float pixel(int x, int y) const;

	int getWidth()const;
	int getHeight() const;
	void getPixels(std::vector<double> *v)const;

	bool errorPars()const;

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

	std::string setTxtFileName(std::string nameInput) {

		std::string nameOfImage = nameInput;
		std::string txtFileName;
		std::string point = ".";
		int cont = 0;
		for (int i = 0; i < nameOfImage.length(); i++) {
			if (nameOfImage.at(i) == point.at(0)) {
				txtFileName = nameOfImage.substr(0, i);
				txtFileName += ".txt";
				break;
			}
		}
		return txtFileName;
	}
};


#endif _IMAGE_UTILITY_


