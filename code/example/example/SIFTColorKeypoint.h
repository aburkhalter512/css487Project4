#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"

#include <limits>

namespace cv
{
	class SIFTColorKeypoint
	{
	public:
		//! the default constructor
		CV_WRAP SIFTColorKeypoint() : pt(0, 0), color(0, 0, 0), response(0), octave(0), class_id(-1) {}
		//! the full constructor
		SIFTColorKeypoint(Point2f _pt, Vec3b _color,
			float _response = 0, int _octave = 0, int _class_id = -1)
			: pt(_pt), color(_color),
			response(_response), octave(_octave), class_id(_class_id) {}
		//! another form of the full constructor
		CV_WRAP SIFTColorKeypoint(float x, float y, Vec3b _color,
			float _response = 0, int _octave = 0, int _class_id = -1)
			: pt(x, y), color(_color),
			response(_response), octave(_octave), class_id(_class_id) {}

		~SIFTColorKeypoint();

		// CV_PROP_RW Macro used as purely readable syntatic sugar
		CV_PROP_RW Point2f pt; //!< coordinates of the keypoints
		CV_PROP_RW float size;
		CV_PROP_RW Vec3b color; //!< keypoint color used for comparison
		CV_PROP_RW float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
		CV_PROP_RW int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
		CV_PROP_RW int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)
	};
}
