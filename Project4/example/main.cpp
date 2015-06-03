// main.cpp
// Authors: Clark Olson, Nick Huebner, James Timmerman

#include "DescriptorUtil.h"
#include "DescriptorType.h"
#include "ScriptData.h"
#include "logging.h"
#include "ColorDescriptorExtractor.h"

#include <iostream>
#include <Windows.h> // Just for creating the data directory

#ifdef _DEBUG
#include "pugixml.hpp"
#include <time.h>
#include <string>
#endif

using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
	DescriptorUtil descriptorUtil;

	if (argc == 1) {

		// Custom pre-built command line args
#ifdef _DEBUG
#if 0
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
		argv[3] = "flower.png";
		argv[4] = "flower.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to1.txt";
#elif 0
		// Used to test an abscense of a homography file
		argv = new char*[8];
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "flower.jpg";
		argv[4] = "flowerHalfTransform.jpg";
		argv[5] = "1";
		argv[6] = "SIFT";
		argv[7] = "1toHalfTransform.txt";
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
#elif 1
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "flowerBasket.jpg";
		argv[4] = "flowerBasket45rot.jpg";
		argv[5] = "1";
		argv[6] = "SIFT";
		argv[7] = "1to-45rot.txt";
#elif 0
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "circle.png";
		argv[4] = "circle40x40translation.png";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to40x40translation.txt";
#elif 1
		argv[0] = "example.exe";
		argv[1] = "../images/test/";
		argv[2] = "2";
		argv[3] = "flowerBasket.jpg";
		argv[4] = "flowerBasket1.5scale.jpg";
		argv[5] = "1";
		argv[6] = "COLORSIFT";
		argv[7] = "1to1.5scale.txt";
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
	if (!data.failed)
	{
#ifdef _DEBUG
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		localtime_s(&tstruct, &now);
		strftime(buf, sizeof(buf), "%Y-%m-%d--%H-%M-%S", &tstruct);

		string folderName = "COLORSIFT-";
		folderName += buf;

		CreateDirectory(folderName.c_str(), NULL);
#endif
		// Initialize storage
		Mat *images = new Mat[data.numImgs];
		vector<KeyPoint> *kpts = new vector<KeyPoint>[data.numImgs];
		Mat **descriptors = new Mat*[data.numTypes];

		for (int i = 0; i < data.numTypes; ++i)
			descriptors[i] = new Mat[data.numImgs];

		// Load images and compute keypoints for each image
		for (int i = 0; i < data.numImgs; ++i) {
			images[i] = cvLoadImage((data.relativePath + data.imageNames[i]).c_str());
			cout << ">> Computing keypoints for " << data.relativePath + data.imageNames[i] << "..." << endl;

			// Load from file or detect new features
			descriptorUtil.detectFeatures(images[i], kpts[i]);

#ifdef _DEBUG //Save Keypoint information ONLY IF the first descriptor type is COLORSIFT
			if (data.types[i].first != COLOR_DESCR)
				continue;

			pugi::xml_document doc;
			pugi::xml_node root = doc.append_child("root");
			pugi::xml_node xmlKeypoints = root.append_child("keypoints");
			pugi::xml_node xmlKeypoint;

			LOG("========== Keypoints ==========");

			for (size_t j = 0; j < kpts[i].size(); j++)
			{
				xmlKeypoint = xmlKeypoints.append_child("keypoint");
				xmlKeypoint.append_attribute("point").set_value((std::to_string(kpts[i][j].pt.x) + ", " + std::to_string(kpts[i][j].pt.y)).c_str());
				xmlKeypoint.append_attribute("angle").set_value(kpts[i][j].angle);
				xmlKeypoint.append_attribute("size").set_value(kpts[i][j].size);
				xmlKeypoint.append_attribute("octave").set_value(kpts[i][j].octave);

				LOG("\n  Keypoint  " << j << 
					"\n    Point:  " << kpts[i][j].pt <<
					"\n    Angle:  " << kpts[i][j].angle <<
					"\n    Size:   " << kpts[i][j].size <<
					"\n    Octave: " << kpts[i][j].octave);
			}

			doc.save_file((folderName + "\\Keypoints-" + data.imageNames[i] + ".xml").c_str());

			std::cout << "Press enter to continue..." << std::endl;
			std::cin.get();
#endif
		}
		cout << ">> Finished computing all keypoints" << endl;

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

#ifdef _DEBUG // Save descriptor information only if COLORSIFT is the data type
					if (data.types[i].first != COLOR_DESCR)
						continue;

					pugi::xml_document doc;
					pugi::xml_node root = doc.append_child("root");
					pugi::xml_node xmlDescriptors = root.append_child("descriptors");
					pugi::xml_node xmlDescriptor;
					pugi::xml_node xmlKeypoint;
					pugi::xml_node xmlDescriptorColor;

					for (size_t descriptorIndex = 0; descriptorIndex < kpts[j].size(); descriptorIndex++)
					{
						LOG("========== Descriptor ==========");
						xmlDescriptor = xmlDescriptors.append_child("descriptor");
						string colorHistogram;
						string xmlColorHistogram;
						int index = 0;

						xmlKeypoint = xmlDescriptor.append_child("keypoint");
						xmlKeypoint.append_attribute("point").set_value((std::to_string(kpts[j][descriptorIndex].pt.x) + ", " + std::to_string(kpts[i][j].pt.y)).c_str());
						xmlKeypoint.append_attribute("angle").set_value(kpts[j][descriptorIndex].angle);
						xmlKeypoint.append_attribute("size").set_value(kpts[j][descriptorIndex].size);
						xmlKeypoint.append_attribute("octave").set_value(kpts[j][descriptorIndex].octave);
						for (int histRow = 0; histRow < ColorDescriptorExtractor::COLOR_DESCR_WIDTH; histRow++)
						{
							for (int histCol = 0; histCol < ColorDescriptorExtractor::COLOR_DESCR_WIDTH; histCol++)
							{
								LOG("  Hist row " << histRow << ", Hist col " << histCol);
								xmlDescriptorColor = xmlDescriptor.append_child("colorHistogram");
								colorHistogram = "\n";
								xmlColorHistogram = "";

								for (int rColor = 0; rColor < ColorDescriptorExtractor::COLOR_DESCR_HIST_BINS; rColor++)
								{
									for (int gColor = 0; gColor < ColorDescriptorExtractor::COLOR_DESCR_HIST_BINS; gColor++)
									{
										for (int bColor = 0; bColor < ColorDescriptorExtractor::COLOR_DESCR_HIST_BINS; bColor++)
										{
											colorHistogram +=
												string("    RGB(") + std::to_string(rColor) + ", " +
												std::to_string(gColor) + ", " +
												std::to_string(bColor) + "): " +
												std::to_string(descriptors[i][j].ptr<float>(descriptorIndex)[index]) + "\n";
											xmlColorHistogram +=
												string("(") + std::to_string(rColor) + "," +
												std::to_string(gColor) + "," +
												std::to_string(bColor) + "): " +
												std::to_string(descriptors[i][j].ptr<float>(descriptorIndex)[index]) + ", ";

											index++;
										}
									}
								}
								xmlDescriptorColor.text().set(xmlColorHistogram.c_str());
								xmlDescriptorColor.append_attribute("row").set_value(histRow);
								xmlDescriptorColor.append_attribute("col").set_value(histCol);

								LOG("  Descriptor Histogram: " << colorHistogram);
							}
						}
					}

					doc.save_file((folderName + "\\Descriptor-" + data.imageNames[j] + ".xml").c_str());

					std::cout << "Press enter to continue..." << std::endl;
					std::cin.get();
#endif
				}
			}
		}

		cout << ">> Finished creating all descriptors." << endl;

		bool drawMatches = true;
		// Matching using homographies, if provided
		if (data.homographyFlag) {
			for (int i = 0; i < data.numImgs - 1; ++i) {
				for (int j = 0; j < data.numTypes; ++j) {
					stringstream outFilename;
					outFilename << folderName << "\\Match_" << j << "_Base_" << data.imageNames[0] << "_Compare_" << data.imageNames[i + 1] << ".txt";
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