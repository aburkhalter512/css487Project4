#include "ColorDescriptorExtractor.h"
#include "logging.h"

#define IMAGE_DATA_TYPE CV_8UC3
#define RED 2
#define GREEN 1
#define BLUE 0
#define INDEX_3D_TO_1D(i, j, k, dimensionSize) i + dimensionSize * j + dimensionSize * dimensionSize * k
#define VEC3_MAG(r, g, b) sqrt(r*r + g*g + b*b)

namespace cv
{
	CV_INIT_ALGORITHM(ColorDescriptorExtractor, "Feature2D.ColorDescriptorExtractor",
		obj.info()->addParam(obj, "nFeatures", obj.nfeatures);
		obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
		obj.info()->addParam(obj, "contrastThreshold", obj.contrastThreshold);
		obj.info()->addParam(obj, "edgeThreshold", obj.edgeThreshold);
		obj.info()->addParam(obj, "sigma", obj.sigma))

	/******************************* Defs and macros *****************************/

	// assumed gaussian blur for input image
	static const float COLOR_INIT_SIGMA = 0.5f;

	// determines the size of a single descriptor orientation histogram
	static const float COLOR_DESCR_SCL_FCTR = 3.f;

	// threshold on magnitude of elements of descriptor vector
	static const float COLOR_DESCR_MAG_THR = 0.2f;

	// factor used to convert floating-point descriptor to unsigned char
	static const float COLOR_INT_DESCR_FCTR = 512.f;

	static const int COLOR_FIXPT_SCALE = 1;

	static inline void
		unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
	{
		octave = kpt.octave & 255;
		layer = (kpt.octave >> 8) & 255;
		octave = octave < 128 ? octave : (-128 | octave);
		scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
	}

	// Takes an image loaded in a matrix and returns its equivalent CV_32FC3 matrix version
	// Essentially it converts it to a 3 channel matrix if not already, and then converts
	// the internal data to 32 bit depth float (for each channel).
	static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
	{
		Mat color, colorFloatingPt;
		if (img.channels() == 1)
			cvtColor(img, color, CV_GRAY2RGB);
		else
			img.copyTo(color);
		color.convertTo(colorFloatingPt, IMAGE_DATA_TYPE);

		float sig_diff;

		if (doubleImageSize)
		{
			sig_diff = sqrtf(std::max(sigma * sigma - COLOR_INIT_SIGMA * COLOR_INIT_SIGMA * 4, 0.01f));
			Mat dbl;
			resize(colorFloatingPt, dbl, Size(color.cols * 2, color.rows * 2), 0, 0, INTER_LINEAR);
			GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
			return dbl;
		}
		else
		{
			sig_diff = sqrtf(std::max(sigma * sigma - COLOR_INIT_SIGMA * COLOR_INIT_SIGMA, 0.01f));
			GaussianBlur(colorFloatingPt, colorFloatingPt, Size(), sig_diff, sig_diff);
			return colorFloatingPt;
		}
	}

	int ColorDescriptorExtractor::descriptorSize() const
	{
		return COLOR_DESCR_WIDTH * COLOR_DESCR_WIDTH * 
			COLOR_DESCR_HIST_BINS * COLOR_DESCR_HIST_BINS * COLOR_DESCR_HIST_BINS;
	}

	int ColorDescriptorExtractor::descriptorType() const
	{
		return IMAGE_DATA_TYPE;
	}

	void ColorDescriptorExtractor::calcColorDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst) const
	{
		// Makes the point have only integer values
		Point pt(cvRound(ptf.x), cvRound(ptf.y));

		// These are the trig values used to rotate the image for rotation invariance
		float cos_t = cosf(ori*(float)(CV_PI / 180));
		float sin_t = sinf(ori*(float)(CV_PI / 180));

		// Gaussian weighting scale for taking the weights of the colors in the descriptor histograms
		float exp_scale = -1.f / (d * d * 0.5f);

		float hist_width = COLOR_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);

		// Used to determine which color will fall in which buckets
		float bucketColor[3] = { 0, 0, 0 };
		
		// The code expects n to be the total descriptor size, so create a dimsize variable
		// to hold the current n. Also create some variables for highly used values.
		int dimSize = n; 
		int sqrDimSize = n * n;
		n *= sqrDimSize;
		int histSize = n * d * d;

		int bufferD = d + 2;
		int bufferN = n + 2;

		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int) sqrt((double) img.cols * img.cols + img.rows * img.rows));

		cos_t /= hist_width;
		sin_t /= hist_width;

		int i, j, k,
			len = (radius * 2 + 1)*(radius * 2 + 1);
		int rows = img.rows, 
			cols = img.cols;

		float *W = new float[len],
			*RColor = new float[len],
			*GColor = new float[len],
			*BColor = new float[len],
			*CMag = new float[len],
			*RBin = new float[len],
			*CBin = new float[len],
			*hist = new float[len];

		// Set all hist values to 0
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
				for (k = 0; k < n; k++)
					hist[(i * d + j) * n + k] = 0.0f;

		for (i = -radius, k = 0; i <= radius; i++)
		{
			for (j = -radius; j <= radius; j++)
			{
				// Calculate sample's histogram array coords rotated relative to ori.
				// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
				// r_rot = 1.5) have full weight placed in row 1 after interpolation.

				// 2d matrix rotation
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;

				// Get the r and c bins based on the rotated indices above
				float rbin = r_rot + d / 2 - 0.5f;
				float cbin = c_rot + d / 2 - 0.5f;

				int r = pt.y + i,
					c = pt.x + j;

				if (rbin > -1 && rbin < d &&
					cbin > -1 && cbin < d &&
					r > 0 && r < rows - 1 &&
					c > 0 && c < cols - 1)
				{
					RBin[k] = rbin;
					CBin[k] = cbin;

					// "Smooth" the weights based on distance from the center
					W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;

					// Put smoothed pixel color values into correct buckets
					for (int color = BLUE; color <= RED; color++)
					{
						bucketColor[color] = (float)
							(.5  * img.at<Vec3b>(r, c)[color]
							+ .25 * img.at<Vec3b>(r, c)[color == BLUE ? RED : color - 1]
							+ .25 * img.at<Vec3b>(r, c)[color == RED ? BLUE : color + 1]);
					}
					
					//// Non-smoothed version
					//for (int color = BLUE; color <= RED; color++)
					//	bucketColor[color] = (float) img.at<Vec3b>(r, c)[color];
					
					// TODO: Remove or fix this
					CMag[k] = VEC3_MAG(bucketColor[RED] / 256, bucketColor[GREEN] / 256, bucketColor[BLUE] / 256);

					RColor[k] = bucketColor[RED] * dimSize / 256;
					GColor[k] = bucketColor[GREEN] * dimSize / 256;
					BColor[k] = bucketColor[BLUE] * dimSize / 256;

					k++;
				}
			}
		}

		len = k;
		exp(W, W, len);

		// Non-interpolation version of the histogram accumulation
		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k];
			float cbin = CBin[k];
			float colorBin = INDEX_3D_TO_1D(RColor[k], GColor[k], BColor[k], dimSize);

			int rowBin = cvFloor(rbin);
			int colBin = cvFloor(cbin);

			int index = (rowBin * d + colBin) * n + colorBin;

			// Bound index to the valid array values
			if (index < 0 || index >= histSize)
				continue;

			dst[index] += 1;
		}

		delete[] W;
		delete[] RColor;
		delete[] GColor;
		delete[] BColor;
		delete[] CMag;
		delete[] RBin;
		delete[] CBin;
		delete[] hist;

		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = d * d * n;
		for (k = 0; k < len; k++)
			nrm2 += dst[k] * dst[k];

		float thr = std::sqrt(nrm2) * COLOR_DESCR_MAG_THR;
		for (i = 0, nrm2 = 0; i < k; i++)
		{
			float val = std::min(dst[i], thr);
			dst[i] = val;
			nrm2 += val * val;
		}
		nrm2 = COLOR_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

		for (k = 0; k < len; k++)
			dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
	}

	void ColorDescriptorExtractor::calcColorDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
		Mat& descriptors, int nOctaveLayers, int firstOctave) const
	{
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale;
			unpackOctave(kpt, octave, layer, scale);

			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);

			float size = kpt.size*scale;
			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
			const Mat& img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer];

			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;

			LOG("Creating descriptor...");
			float* descriptor = descriptors.ptr<float>((int) i);
			calcColorDescriptor(img, ptf, angle, 
				size * 0.5f, COLOR_DESCR_WIDTH, COLOR_DESCR_HIST_BINS,
				descriptor);
			LOG("Done.");
		}
	}

	void ColorDescriptorExtractor::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const
	{
		int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
		Mat image = _image.getMat(), mask = _mask.getMat();

		if (image.empty() || image.depth() != CV_8U)
			CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

		if (!mask.empty() && mask.type() != CV_8UC1)
			CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

		if (useProvidedKeypoints)
		{
			firstOctave = 0;
			int maxOctave = INT_MIN;

			for (size_t i = 0; i < keypoints.size(); i++)
			{
				int octave, layer;
				float scale;
				unpackOctave(keypoints[i], octave, layer, scale);

				firstOctave = std::min(firstOctave, octave);
				maxOctave = std::max(maxOctave, octave);

				actualNLayers = std::max(actualNLayers, layer - 2);
			}

			firstOctave = std::min(firstOctave, 0);

			CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);

			actualNOctaves = maxOctave - firstOctave + 1;
		}

		Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
		vector<Mat> gpyr, dogpyr;
		int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);

		if (!useProvidedKeypoints)
		{
			findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				KeyPointsFilter::retainBest(keypoints, nfeatures);

			if (firstOctave < 0)
				for (size_t i = 0; i < keypoints.size(); i++)
				{
					KeyPoint& kpt = keypoints[i];
					float scale = 1.f / (float)(1 << -firstOctave);
					kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
					kpt.pt *= scale;
					kpt.size *= scale;
				}

			if (!mask.empty())
				KeyPointsFilter::runByPixelsMask(keypoints, mask);
		}

		if (_descriptors.needed())
		{
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();
			descriptors = Scalar::all(0);

			calcColorDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
		}
	}
}