#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"

#include <limits>

namespace cv
{
	class ColorKeypoint
	{
	public:
		//! the default constructor
		CV_WRAP ColorKeypoint() : pt(0, 0), size(0), angle(0), color(0, 0, 0), response(0), octave(0), class_id(-1) {}




		//! the full constructor
		CV_WRAP ColorKeypoint(Point2f _pt, float _size, float _angle = 0, Vec3b _color = Vec3b(0, 0, 0), float _response = 0, int _octave = 0, int _class_id = -1)
			: pt(_pt), size(_size), angle(_angle), color(_color), response(_response), octave(_octave), class_id(_class_id) {};
		
		//! x, y instead of pt
		CV_WRAP ColorKeypoint(float x, float y, float _size, float _angle = 0, Vec3b _color = Vec3b(0, 0, 0), float _response = 0, int _octave = 0, int _class_id = -1)
			: pt(x, y), size(_size), angle(_angle), color(_color), response(_response), octave(_octave), class_id(_class_id) {};


		//METHODS
		size_t hash() const;

		void convert(const std::vector<ColorKeypoint>& keypoints, std::vector<Point2f>& points2f,
			const vector<int>& keypointIndexes);

		void convert(const std::vector<Point2f>& points2f, std::vector<ColorKeypoint>& keypoints,
			float size, float response, int octave, int class_id);

		float overlap(const ColorKeypoint& kp1, const ColorKeypoint& kp2);

		
		

		class  MaskPredicate;
		struct KeypointResponseGreaterThanThreshold;
		struct KeypointResponseGreater;
		struct RoiPredicate;
		struct SizePredicate;
		struct KeyPoint_LessThan;


		// CV_PROP_RW Macro used as purely readable syntatic sugar
		CV_PROP_RW Point2f pt; //!< coordinates of the keypoints
		CV_PROP_RW float size;
		CV_PROP_RW float angle;
		CV_PROP_RW Vec3b color; //!< keypoint color used for comparison
		CV_PROP_RW float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
		CV_PROP_RW int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
		CV_PROP_RW int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)
	};

	class ColorKeyPointsFilter {
	public:
		static void retainBest(vector<ColorKeypoint>& keypoints, int n_points);
		static void runByImageBorder(vector<ColorKeypoint>& keypoints, Size imageSize, int borderSize);
		static void runByKeypointSize(vector<ColorKeypoint>& keypoints, float minSize, float maxSize);
		static void runByPixelsMask(vector<ColorKeypoint>& keypoints, const Mat& mask);
		static void removeDuplicated(vector<ColorKeypoint>& keypoints);
	};
}
