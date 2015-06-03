#include "ColorDescriptorExtractor.h"
#include "logging.h"

#ifdef _DEBUG
#include "pugixml.hpp"
#include <time.h>
#include <string>
#endif

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

	// default width of descriptor histogram array
	static const int COLOR_DESCR_WIDTH = 4;

	// default number of bins per histogram in descriptor array
	static const int COLOR_DESCR_HIST_BINS = 2;

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

#ifdef _DEBUG
	void ColorDescriptorExtractor::keypointsFile(string filename)
	{
		_keypointsFilename = filename;
	}
	void ColorDescriptorExtractor::descriptorsFile(string filename)
	{
		_descriptorsFilename = filename;
	}
#endif

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
			len = (radius * 2 + 1)*(radius * 2 + 1), 
			histlen = bufferD * bufferD * bufferN;
		int rows = img.rows, 
			cols = img.cols;

		// Just a mass memory allocation
		/*AutoBuffer<float> buf(len * 8);
		float *W = buf,
			*RColor = W + len,
			*GColor = RColor + len,
			*BColor = GColor + len,
			*CMag = BColor + len,
			*RBin = CMag + len, 
			*CBin = RBin + len, 
			*hist = CBin + len;*/

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

					//// Put smoothed pixel color values into correct buckets
					//for (int color = RED; color <= BLUE; color++)
					//{
					//	bucketColor[color] = (float)
					//		(.5  * img.at<Vec3f>(r, c)[color]
					//		* .25 * img.at<Vec3f>(r, c)[color == RED ? BLUE : color - 1]
					//		* .25 * img.at<Vec3f>(r, c)[color == BLUE ? RED : color + 1]);
					//}

					
					// Non-smoothed version
					for (int color = BLUE; color <= RED; color++)
						bucketColor[color] = (float) img.at<Vec3b>(r, c)[color];
					
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

		//for (k = 0; k < len; k++)
		//{
		//	float rbin = RBin[k], 
		//		cbin = CBin[k];

		//	// The whole trilinear interpolation thing needs to be figured out... I really
		//	// don't think it is right in its current state.
		//	float colorBin = INDEX_3D_TO_1D(RColor[k], GColor[k], BColor[k], dimSize);
		//	float mag = /*CMag[k] * */W[k];

		//	int r0 = cvFloor(rbin);
		//	int c0 = cvFloor(cbin);
		//	int color0 = cvFloor(colorBin);

		//	// Calculate the 
		//	rbin -= r0;
		//	cbin -= c0;
		//	colorBin -= color0;

		//	// histogram update using tri-linear interpolation
		//	float v_r1 = /*mag**/rbin, v_r0 = mag - v_r1;
		//	float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		//	float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;

		//	float v_rco111 = v_rc11 * colorBin;
		//	float v_rco110 = v_rc11 - v_rco111;
		//	float v_rco101 = v_rc10 * colorBin;
		//	float v_rco100 = v_rc10 - v_rco101;
		//	float v_rco011 = v_rc01 * colorBin;
		//	float v_rco010 = v_rc01 - v_rco011;
		//	float v_rco001 = v_rc00 * colorBin;
		//	float v_rco000 = v_rc00 - v_rco001;

		//	int idx = ((r0 + 1)*(bufferD) + c0 + 1)*(bufferN) + color0;
		//	hist[idx] += v_rco000;
		//	hist[idx + 1] += v_rco001;
		//	hist[idx + (bufferN)] += v_rco010;
		//	hist[idx + (bufferN + 1)] += v_rco011;
		//	hist[idx + (bufferD)*(bufferN)] += v_rco100;
		//	hist[idx + (bufferD)*(bufferN) + 1] += v_rco101;
		//	hist[idx + (bufferD + 1)*(bufferN)] += v_rco110;
		//	hist[idx + (bufferD + 1)*(bufferN) + 1] += v_rco111;
		//}

		//// finalize histogram, since the orientation histograms are circular
		//for (i = 0; i < d; i++)
		//{
		//	for (j = 0; j < d; j++)
		//	{
		//		int idx = ((i + 1) * bufferD + (j + 1)) * bufferN;
		//		hist[idx] += hist[idx + n];
		//		hist[idx + 1] += hist[idx + n + 1];
		//		for (k = 0; k < n; k++)
		//			dst[(i * d + j) * n + k] = hist[idx + k];
		//	}
		//}

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
#ifdef _DEBUG
		pugi::xml_document doc;
		pugi::xml_node root = doc.append_child("root");
		pugi::xml_node xmlDescriptors = root.append_child("descriptors");
		pugi::xml_node timing = root.append_child("timing");
		pugi::xml_node xmlDescriptor;
		pugi::xml_node timingMark;
		int64 startTime = getTickCount();
#endif

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

#ifdef _DEBUG
			timingMark = timing.append_child("mark");
			timingMark.append_attribute("metric").set_value("second");
			timingMark.append_attribute("time").set_value((float)(getTickCount() - startTime) / getTickFrequency());

			LOG("========== Descriptor ==========");
			xmlDescriptor = xmlDescriptors.append_child("descriptor");
			string colorHistogram;
			string xmlColorHistogram;
			int index = 0;
			for (int histRow = 0; histRow < COLOR_DESCR_WIDTH; histRow++)
			{
				for (int histCol = 0; histCol < COLOR_DESCR_WIDTH; histCol++)
				{
					LOG("  Hist row " << histRow << ", Hist col " << histCol);
					pugi::xml_node xmlDescriptorColor = xmlDescriptor.append_child("colorHistogram");
					colorHistogram = "\n";
					xmlColorHistogram = "";

					for (int rColor = 0; rColor < COLOR_DESCR_HIST_BINS; rColor++)
					{
						for (int gColor = 0; gColor < COLOR_DESCR_HIST_BINS; gColor++)
						{
							for (int bColor = 0; bColor < COLOR_DESCR_HIST_BINS; bColor++)
							{
								colorHistogram += 
									string("    RGB(") + std::to_string(rColor) + ", " + 
									std::to_string(gColor) + ", " + 
									std::to_string(bColor) + "): " +
									std::to_string(descriptor[index]) + "\n";
								xmlColorHistogram +=
									string("(") + std::to_string(rColor) + "," +
									std::to_string(gColor) + "," +
									std::to_string(bColor) + "): " +
									std::to_string(descriptor[index]) + ", ";

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
#endif
		}

#ifdef _DEBUG
		std::cout << "Press enter to continue..." << std::endl;
		std::cin.get();

		doc.save_file(_descriptorsFilename.c_str());
#endif
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

		//double t, tf = getTickFrequency();
		//t = (double)getTickCount();
		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);

		//t = (double)getTickCount() - t;
		//printf("pyramid construction time: %g\n", t*1000./tf);

		if (!useProvidedKeypoints)
		{
			//t = (double)getTickCount();
			findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				KeyPointsFilter::retainBest(keypoints, nfeatures);
			//t = (double)getTickCount() - t;
			//printf("keypoint detection time: %g\n", t*1000./tf);

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

#ifdef _DEBUG
		LOG("========== Keypoints ==========");

		for (size_t i = 0; i < keypoints.size(); i++)
		{
			LOG("  Keypoint  " << i);
			LOG("    Point:  " << keypoints[i].pt);
			LOG("    Angle:  " << keypoints[i].angle);
			LOG("    Size:   " << keypoints[i].size);
			LOG("    Octave: " << keypoints[i].octave);
		}

		std::cout << "Press enter to continue..." << std::endl;
		std::cin.get();
#endif

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();
			descriptors = Scalar::all(0);

			calcColorDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}
}