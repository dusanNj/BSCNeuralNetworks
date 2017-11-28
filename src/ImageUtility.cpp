#include "ImageUtility.h"



ImageUtility::ImageUtility(std::string file, int targetSize, bool boundingBox)
{
	Mat *imagGray;
	imagGray = &imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	if (imagGray->empty()) {
		error = true;
		return;
	}

	if (imagGray->cols * imagGray->rows != NORM_HEIGHT*NORM_WIDTH) {
		error = true;
		return;
	}

	threshold(*imagGray, *imagGray, 0, 255, CV_THRESH_BINARY);

	pixels = new float[imagGray->cols*imagGray->rows];

	for (int i = 0; i < imagGray->cols; i++) {
		for (int j = 0; j < imagGray->rows; j++) {
			pixels[i*imagGray->cols + j] = imagGray->at<uchar>(i, j);
		}

	}

}


ImageUtility::~ImageUtility()
{
	free(pixels);
}

int ImageUtility::getWidth() const {
	return width;
}

int ImageUtility::getHeight() const {
	return height;
}

void ImageUtility::getPixels(std::vector<double> *v) const {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*v)[i*width + j] = pixels[i*width + j];
		}
	}
}

float ImageUtility::pixel(int x, int y) const {
	if (x < 0 || x >= NORM_WIDTH ||
		y < 0 || y >= NORM_HEIGHT) {
		return NULL;
	}
	return pixels[y * width + x];
}