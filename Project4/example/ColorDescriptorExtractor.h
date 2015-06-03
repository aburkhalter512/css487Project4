#pragma once
#include "NewDescriptorExtractor.h"

namespace cv
{
	class ColorDescriptorExtractor : public NEWSIFT
	{
	public:
		CV_WRAP explicit ColorDescriptorExtractor(int nfeatures = 0, int nOctaveLayers = 3,
			double contrastThreshold = 0.04, double edgeThreshold = 10,	double sigma = 1.6)
			: NEWSIFT(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma) {}

		//! returns the descriptor size in floats (128)
		CV_WRAP int descriptorSize() const;
		CV_WRAP int descriptorType() const;

		//! finds the keypoints using SIFT algorithm
		void operator()(InputArray img, InputArray mask,
			vector<KeyPoint>& keypoints) const
		{
			(*this)(img, mask, keypoints, noArray(), false);
		}
		void operator()(InputArray _image, InputArray _mask, 
			vector<KeyPoint>& keypoints, OutputArray _descriptors, 
			bool useProvidedKeypoints) const;

		AlgorithmInfo* info() const;

#ifdef _DEBUG
		void keypointsFile(string filename);
		void descriptorsFile(string filename);
#endif

	private:
		void calcColorDescriptor(const Mat&, Point2f, float, float, int, int, float*) const;
		void calcColorDescriptors(const vector<Mat>&, const vector<KeyPoint>&, Mat&, int, int) const;

		void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
		{
			(*this)(image, Mat(), keypoints, descriptors, true);
		}

#ifdef _DEBUG
		string _keypointsFilename;
		string _descriptorsFilename;
#endif
	};
}
