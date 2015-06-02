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

		void operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const;

	private:
		void calcColorDescriptor(const Mat&, Point2f, float, float, int, int, float*) const;
		void calcColorDescriptors(const vector<Mat>&, const vector<KeyPoint>&, Mat&, int, int) const;
	};
}
