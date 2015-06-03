// main.cpp
// Authors: Clark Olson, Nick Huebner, James Timmerman

#include "DescriptorUtil.h"
#include "DescriptorType.h"
#include "ScriptData.h"
#include "logging.h"

#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
	DescriptorUtil descriptorUtil;

	if (argc == 1) {

#ifdef _DEBUG
#if 1
		// Used a good set of sample parameters
		argv = new char*[8];
		argv[0] = "example.exe";
		argv[1] = "../images/bark/";
		argv[2] = "2";
		argv[3] = "img1.ppm";
		argv[4] = "img2.ppm";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "H1to2p.txt";
#elif 0
		// Used to test an abscense of a homography file
		argv = new char*[8];
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "fireFlower.png";
		argv[4] = "fireFlower.png";
		argv[5] = "1";
		argv[6] = "SIFT";
		argv[7] = "1to1.txt";
#elif 0
		// Used to test an abscense of a homography file
		argv = new char*[8];
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "circle.png";
		argv[4] = "circle.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to1.txt";
#elif 0
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "circle.png";
		argv[4] = "circle45cclock.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to45cClock.txt";
#elif 1
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "circle.png";
		argv[4] = "circle40x40translation.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to40x40translation.txt";
#elif 0
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "circle.png";
		argv[4] = "circle1.05scale.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to1.05scale.txt";
#endif
#else
		argv = new char*[8];
		argv[0] = "example.exe";
		argv[1] = "../images/bark/";
		argv[2] = "2";
		argv[3] = "img1.ppm";
		argv[4] = "img2.ppm";
		argv[5] = "1";
		argv[6] = "NEWSIFT";
		argv[7] = "H1to2p.txt";
#endif
	}

	string scriptFilename = argv[1];
	ScriptData data(argv);

	// If the script succeeded in loading
	if (!data.failed) {
		// Initialize storage
		Mat *images = new Mat[data.numImgs];
		vector<KeyPoint> *kpts = new vector<KeyPoint>[data.numImgs];
		Mat **descriptors = new Mat*[data.numTypes];
		for (int i = 0; i < data.numTypes; ++i) {
			descriptors[i] = new Mat[data.numImgs];
		}

		// Load images and compute keypoints for each image
		for (int i = 0; i < data.numImgs; ++i) {
			images[i] = cvLoadImage((data.relativePath + data.imageNames[i]).c_str());
			cout << ">> Computing keypoints for " << data.imageNames[i] << "..." << endl;

			// Load from file or detect new features
			descriptorUtil.detectFeatures(images[i], kpts[i]);
			// kpts[i] = descriptorUtil.readKeyPoints(data.relativePath + "kpts.xml", data.imageNames[i].substr(0, data.imageNames[i].length() - 4 ));
		}
		cout << ">> Finished computing all keypoints" << endl;

		// Save keypoints if save flag is set
		if (data.saveData) {
			cout << ">> Saving keypoints to: kpts.xml" << endl;
			descriptorUtil.writeKeyPoints(kpts, data.imageNames, data.numImgs, data.relativePath + "kpts.xml");
		}

		// Compute descriptors
		for (int i = 0; i < data.numTypes; ++i) {
			// Inner array of descriptor matrices contains only one type of descriptor
			cout << ">> Computing descriptor type #" << i << "..." << endl;

			// Compute descriptors for each image
			for (int j = 0; j < data.numImgs; ++j) {
				if (data.types[i].doubleDescriptor) {
					Mat &descriptors1 = descriptorUtil.computeDescriptors(images[j], kpts[j], data.types[i].first);
					Mat &descriptors2 = descriptorUtil.computeDescriptors(images[j], kpts[j], data.types[i].second);

					// Merge descriptors
					descriptors[i][j] = descriptorUtil.mergeDescriptors(descriptors1, descriptors2);
				}
				else {
					descriptors[i][j] = descriptorUtil.computeDescriptors(images[j], kpts[j], data.types[i].first);
				}
			}
		}

		cout << ">> Finished creating all descriptors." << endl;

		// Save descriptors if save flag is set
		// data.saveData = true;

		if (data.saveData) {
			for (int i = 0; i < data.numTypes; ++i) {
				cout << ">> Saving descriptors for type #" << i << endl;

				stringstream descriptorFilePath;
				descriptorFilePath << data.relativePath << "descriptors" << i << ".xml";
				descriptorUtil.writeDescriptors(descriptors[i], data.imageNames, data.numImgs, descriptorFilePath.str());
			}
		}

		bool drawMatches = true;
		// Matching using homographies, if provided
		if (data.homographyFlag) {
			for (int i = 0; i < data.numImgs - 1; ++i) {
				for (int j = 0; j < data.numTypes; ++j) {
					stringstream outFilename;
					outFilename << data.relativePath << "desc_" << j << "_img_" << (i + 1) << ".txt";
					descriptorUtil.match(descriptors[j][0], descriptors[j][i + 1], kpts[0], kpts[i + 1], images[0], images[i + 1], data.homographies[i], outFilename.str(), drawMatches);
				}
			}
		}

		// Memory cleanup
		delete[] images;
		delete[] kpts;
		for (int i = 0; i < data.numTypes; ++i) {
			delete[] descriptors[i];
		}
		delete[] descriptors;
		if (argc == 1) {
			delete argv;
		}
	}
}