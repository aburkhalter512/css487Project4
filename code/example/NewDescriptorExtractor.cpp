/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "NewDescriptorExtractor.h"
#include <iostream>
#include <stdarg.h>
#include "ColorKeypoint.h"

#define INDEX_3D_TO_1D(i, j, k, dimensionSize) i + dimensionSize * j + dimensionSize * dimensionSize * k

namespace cv
{
	CV_INIT_ALGORITHM(NEWSIFT, "Feature2D.NEWSIFT",
		obj.info()->addParam(obj, "nFeatures", obj.nfeatures);
		obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
		obj.info()->addParam(obj, "contrastThreshold", obj.contrastThreshold);
		obj.info()->addParam(obj, "edgeThreshold", obj.edgeThreshold);
		obj.info()->addParam(obj, "sigma", obj.sigma))

	/******************************* Defs and macros *****************************/

	// default number of sampled intervals per octave
	static const int NEWSIFT_INTVLS = 3;

	// default sigma for initial gaussian smoothing
	static const float NEWSIFT_SIGMA = 1.6f;

	// default threshold on keypoint contrast |D(x)|
	static const float NEWSIFT_CONTR_THR = 0.04f;

	// default threshold on keypoint ratio of principle curvatures
	static const float NEWSIFT_CURV_THR = 10.f;

	// double image size before pyramid construction?
	static const bool NEWSIFT_IMG_DBL = true;

	// default width of descriptor histogram array
	static const int NEWSIFT_DESCR_WIDTH = 4;

	// default number of bins per histogram in descriptor array
	static const int NEWSIFT_DESCR_HIST_BINS = 8;

	// assumed gaussian blur for input image
	static const float NEWSIFT_INIT_SIGMA = 0.5f;

	// width of border in which to ignore keypoints
	static const int NEWSIFT_IMG_BORDER = 5;

	// maximum steps of keypoint interpolation before failure
	static const int NEWSIFT_MAX_INTERP_STEPS = 5;

	// default number of bins in histogram for orientation assignment
	static const int NEWSIFT_ORI_HIST_BINS = 36;

	// determines gaussian sigma for orientation assignment
	static const float NEWSIFT_ORI_SIG_FCTR = 1.5f;

	// determines the radius of the region used in orientation assignment
	static const float NEWSIFT_ORI_RADIUS = 3 * NEWSIFT_ORI_SIG_FCTR;

	// orientation magnitude relative to max that results in new feature
	static const float NEWSIFT_ORI_PEAK_RATIO = 0.8f;

	// determines the size of a single descriptor orientation histogram
	static const float NEWSIFT_DESCR_SCL_FCTR = 3.f;

	// threshold on magnitude of elements of descriptor vector
	static const float NEWSIFT_DESCR_MAG_THR = 0.2f;

	// factor used to convert floating-point descriptor to unsigned char
	static const float NEWSIFT_INT_DESCR_FCTR = 512.f;

#if 0
	// intermediate type used for DoG pyramids
	typedef short NEWSIFT_wt;
	static const int NEWSIFT_FIXPT_SCALE = 48;
#else
	// intermediate type used for DoG pyramids
	typedef float NEWSIFT_wt;
	static const int NEWSIFT_FIXPT_SCALE = 1;
#endif

	static inline void
		unpackOctave(const ColorKeypoint& kpt, int& octave, int& layer, float& scale)
	{
		octave = kpt.octave & 255;
		layer = (kpt.octave >> 8) & 255;
		octave = octave < 128 ? octave : (-128 | octave);
		scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
	}

	static inline void
		unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
	{
		octave = kpt.octave & 255;
		layer = (kpt.octave >> 8) & 255;
		octave = octave < 128 ? octave : (-128 | octave);
		scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
	}

	static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
	{
		Mat gray, gray_fpt;
		if (img.channels() == 3 || img.channels() == 4)
			cvtColor(img, gray, COLOR_BGR2GRAY);
		else
			img.copyTo(gray);
		gray.convertTo(gray_fpt, DataType<NEWSIFT_wt>::type, NEWSIFT_FIXPT_SCALE, 0);

		float sig_diff;

		if (doubleImageSize)
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA * 4, 0.01f));
			Mat dbl;
			resize(gray_fpt, dbl, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR);
			GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
			return dbl;
		}
		else
		{
			sig_diff = sqrtf(std::max(sigma * sigma - NEWSIFT_INIT_SIGMA * NEWSIFT_INIT_SIGMA, 0.01f));
			GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
			return gray_fpt;
		}
	}


	void NEWSIFT::buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves) const
	{
		vector<double> sig(nOctaveLayers + 3);
		pyr.resize(nOctaves*(nOctaveLayers + 3));

		// precompute Gaussian sigmas using the following formula:
		//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
		sig[0] = sigma;
		double k = pow(2., 1. / nOctaveLayers);
		for (int i = 1; i < nOctaveLayers + 3; i++)
		{
			double sig_prev = pow(k, (double)(i - 1))*sigma;
			double sig_total = sig_prev*k;
			sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
		}

		for (int o = 0; o < nOctaves; o++)
		{
			for (int i = 0; i < nOctaveLayers + 3; i++)
			{
				Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
				if (o == 0 && i == 0)
					dst = base;
				// base of new octave is halved image from end of previous octave
				else if (i == 0)
				{
					const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
					resize(src, dst, Size(src.cols / 2, src.rows / 2),
						0, 0, INTER_NEAREST);
				}
				else
				{
					const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
					GaussianBlur(src, dst, Size(), sig[i], sig[i]);
				}
			}
		}
	}


	void NEWSIFT::buildDoGPyramid(const vector<Mat>& gpyr, vector<Mat>& dogpyr) const
	{
		int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
		dogpyr.resize(nOctaves*(nOctaveLayers + 2));

		for (int o = 0; o < nOctaves; o++)
		{
			for (int i = 0; i < nOctaveLayers + 2; i++)
			{
				const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
				const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
				Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
				subtract(src2, src1, dst, noArray(), DataType<NEWSIFT_wt>::type);
			}
		}
	}


	/*
	//Mat& img --- source image
	//Point pt --- center point
	//int radius --- radius of keypoint area
	//float sigma --- data on keypoint size and blur level
	//int* hist --- histogram of colors, smoothed around the center pt
	//int bucketCount --- number of buckets for each color in the hist
	*/
	static float calcColorHist(const Mat& img, Point pt, int radius,
		float sigma, float* hist, int bucketCount)
	{
		//step 1: allocate necessary memory/initialize values
		int len = (radius * 2 + 1)*(radius * 2 + 1);
		float gaussScale = -1.f / (2.f * sigma * sigma);
		float *gaussWeight = new float[len];
		const int R = 0;
		const int G = 1;
		const int B = 2;
		int bucketIndex[3] = { 0, 0, 0 };

		//Process each pixel in a radius around the center pixel:
		//	1. Collect color data from each pixel
		//	2. Weight its importance according to distance using a guassian ratio
		for (int i = -radius, k = 0; i <= radius; i++)
		{
			int y = pt.y + i;
			if (y <= 0 || y >= img.rows - 1)
				continue;
			for (int j = -radius; j <= radius; j++)
			{
				int x = pt.x + j;
				if (x <= 0 || x >= img.cols - 1)
					continue;
				
				//put smoothed pixel color values into correct buckets
				for (int color = R; color <= B; color++)
				{
					bucketIndex[color] = (int)
						  (.5  * img.at<Vec3b>(y, x)[color]
						* .25 * img.at<Vec3b>(y, x)[(color - 1) % B]
						* .25 * img.at<Vec3b>(y, x)[(color + 1) % B]
						* bucketCount / 256);
				}

				//make gaussian weight ratios for each pixel in the keypoint area
				gaussWeight[k] = exp((i*i + j*j)*gaussScale);

				//add weighted buckets to 1D histogram
				hist[bucketIndex[R] + bucketCount * bucketIndex[G] + bucketCount * bucketCount * bucketIndex[B]]+= gaussWeight[k];

				k++;
			}
		}

		delete[] gaussWeight;

		float maxval = hist[0];
		for (int i = 1; i < bucketCount; i++)
			maxval = std::max(maxval, hist[i]);

		return maxval;
	}

	// Computes a gradient orientation histogram at a specified pixel
	static float calcOrientationHist(const Mat& img, Point pt, int radius,
		float sigma, float* hist, int n)
	{
		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

		float expf_scale = -1.f / (2.f * sigma * sigma);
		AutoBuffer<float> buf(len * 4 + n + 4);
		float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
		float* temphist = W + len + 2;

		for (i = 0; i < n; i++)
			temphist[i] = 0.f;

		for (i = -radius, k = 0; i <= radius; i++)
		{
			int y = pt.y + i;
			if (y <= 0 || y >= img.rows - 1)
				continue;
			for (j = -radius; j <= radius; j++)
			{
				int x = pt.x + j;
				if (x <= 0 || x >= img.cols - 1)
					continue;

				float dx = (float)(img.at<NEWSIFT_wt>(y, x + 1) - img.at<NEWSIFT_wt>(y, x - 1));
				float dy = (float)(img.at<NEWSIFT_wt>(y - 1, x) - img.at<NEWSIFT_wt>(y + 1, x));

				X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
				k++;
			}
		}

		len = k;

		// compute gradient values, orientations and the weights over the pixel neighborhood
		exp(W, W, len);
		fastAtan2(Y, X, Ori, len, true);
		magnitude(X, Y, Mag, len);

		for (k = 0; k < len; k++)
		{
			int bin = cvRound((n / 360.f)*Ori[k]);
			if (bin >= n)
				bin -= n;
			if (bin < 0)
				bin += n;
			temphist[bin] += W[k] * Mag[k];
		}

		// smooth the histogram
		temphist[-1] = temphist[n - 1];
		temphist[-2] = temphist[n - 2];
		temphist[n] = temphist[0];
		temphist[n + 1] = temphist[1];
		for (i = 0; i < n; i++)
		{
			hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
				(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
				temphist[i] * (6.f / 16.f);
		}

		float maxval = hist[0];
		for (i = 1; i < n; i++)
			maxval = std::max(maxval, hist[i]);

		return maxval;
	}

	static bool adjustLocalExtrema(const vector<Mat>& dog_pyr, ColorKeypoint& kpt, int octv,
		int& layer, int& r, int& c, int nOctaveLayers,
		float contrastThreshold, float edgeThreshold, float sigma)
	{
		const float img_scale = 1.f / (255 * NEWSIFT_FIXPT_SCALE);
		const float deriv_scale = img_scale*0.5f;
		const float second_deriv_scale = img_scale;
		const float cross_deriv_scale = img_scale*0.25f;

		float xi = 0, xr = 0, xc = 0, contr = 0;
		int i = 0;

		for (; i < NEWSIFT_MAX_INTERP_STEPS; i++)
		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];

			Vec3f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);

			float v2 = (float)img.at<NEWSIFT_wt>(r, c) * 2;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dss = (next.at<NEWSIFT_wt>(r, c) + prev.at<NEWSIFT_wt>(r, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1))*cross_deriv_scale;
			float dxs = (next.at<NEWSIFT_wt>(r, c + 1) - next.at<NEWSIFT_wt>(r, c - 1) -
				prev.at<NEWSIFT_wt>(r, c + 1) + prev.at<NEWSIFT_wt>(r, c - 1))*cross_deriv_scale;
			float dys = (next.at<NEWSIFT_wt>(r + 1, c) - next.at<NEWSIFT_wt>(r - 1, c) -
				prev.at<NEWSIFT_wt>(r + 1, c) + prev.at<NEWSIFT_wt>(r - 1, c))*cross_deriv_scale;

			Matx33f H(dxx, dxy, dxs,
				dxy, dyy, dys,
				dxs, dys, dss);

			Vec3f X = H.solve(dD, DECOMP_LU);

			xi = -X[2];
			xr = -X[1];
			xc = -X[0];

			if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
				break;

			if (std::abs(xi) > (float)(INT_MAX / 3) ||
				std::abs(xr) > (float)(INT_MAX / 3) ||
				std::abs(xc) > (float)(INT_MAX / 3))
				return false;

			c += cvRound(xc);
			r += cvRound(xr);
			layer += cvRound(xi);

			if (layer < 1 || layer > nOctaveLayers ||
				c < NEWSIFT_IMG_BORDER || c >= img.cols - NEWSIFT_IMG_BORDER ||
				r < NEWSIFT_IMG_BORDER || r >= img.rows - NEWSIFT_IMG_BORDER)
				return false;
		}

		// ensure convergence of interpolation
		if (i >= NEWSIFT_MAX_INTERP_STEPS)
			return false;

		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];
			Matx31f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);
			float t = dD.dot(Matx31f(xc, xr, xi));

			contr = img.at<NEWSIFT_wt>(r, c)*img_scale + t * 0.5f;
			if (std::abs(contr) * nOctaveLayers < contrastThreshold)
				return false;

			// principal curvatures are computed using the trace and det of Hessian
			float v2 = img.at<NEWSIFT_wt>(r, c)*2.f;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1)) * cross_deriv_scale;
			float tr = dxx + dyy;
			float det = dxx * dyy - dxy * dxy;

			if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
				return false;
		}

		kpt.pt.x = (c + xc) * (1 << octv);
		kpt.pt.y = (r + xr) * (1 << octv);
		kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
		kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
		kpt.response = std::abs(contr);

		return true;
	}

	//
	// Interpolates a scale-space extremum's location and scale to subpixel
	// accuracy to form an image feature. Rejects features with low contrast.
	// Based on Section 4 of Lowe's paper.
	static bool adjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
		int& layer, int& r, int& c, int nOctaveLayers,
		float contrastThreshold, float edgeThreshold, float sigma)
	{
		const float img_scale = 1.f / (255 * NEWSIFT_FIXPT_SCALE);
		const float deriv_scale = img_scale*0.5f;
		const float second_deriv_scale = img_scale;
		const float cross_deriv_scale = img_scale*0.25f;

		float xi = 0, xr = 0, xc = 0, contr = 0;
		int i = 0;

		for (; i < NEWSIFT_MAX_INTERP_STEPS; i++)
		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];

			Vec3f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);

			float v2 = (float)img.at<NEWSIFT_wt>(r, c) * 2;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dss = (next.at<NEWSIFT_wt>(r, c) + prev.at<NEWSIFT_wt>(r, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1))*cross_deriv_scale;
			float dxs = (next.at<NEWSIFT_wt>(r, c + 1) - next.at<NEWSIFT_wt>(r, c - 1) -
				prev.at<NEWSIFT_wt>(r, c + 1) + prev.at<NEWSIFT_wt>(r, c - 1))*cross_deriv_scale;
			float dys = (next.at<NEWSIFT_wt>(r + 1, c) - next.at<NEWSIFT_wt>(r - 1, c) -
				prev.at<NEWSIFT_wt>(r + 1, c) + prev.at<NEWSIFT_wt>(r - 1, c))*cross_deriv_scale;

			Matx33f H(dxx, dxy, dxs,
				dxy, dyy, dys,
				dxs, dys, dss);

			Vec3f X = H.solve(dD, DECOMP_LU);

			xi = -X[2];
			xr = -X[1];
			xc = -X[0];

			if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
				break;

			if (std::abs(xi) > (float)(INT_MAX / 3) ||
				std::abs(xr) > (float)(INT_MAX / 3) ||
				std::abs(xc) > (float)(INT_MAX / 3))
				return false;

			c += cvRound(xc);
			r += cvRound(xr);
			layer += cvRound(xi);

			if (layer < 1 || layer > nOctaveLayers ||
				c < NEWSIFT_IMG_BORDER || c >= img.cols - NEWSIFT_IMG_BORDER ||
				r < NEWSIFT_IMG_BORDER || r >= img.rows - NEWSIFT_IMG_BORDER)
				return false;
		}

		// ensure convergence of interpolation
		if (i >= NEWSIFT_MAX_INTERP_STEPS)
			return false;

		{
			int idx = octv*(nOctaveLayers + 2) + layer;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx - 1];
			const Mat& next = dog_pyr[idx + 1];
			Matx31f dD((img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1))*deriv_scale,
				(img.at<NEWSIFT_wt>(r + 1, c) - img.at<NEWSIFT_wt>(r - 1, c))*deriv_scale,
				(next.at<NEWSIFT_wt>(r, c) - prev.at<NEWSIFT_wt>(r, c))*deriv_scale);
			float t = dD.dot(Matx31f(xc, xr, xi));

			contr = img.at<NEWSIFT_wt>(r, c)*img_scale + t * 0.5f;
			if (std::abs(contr) * nOctaveLayers < contrastThreshold)
				return false;

			// principal curvatures are computed using the trace and det of Hessian
			float v2 = img.at<NEWSIFT_wt>(r, c)*2.f;
			float dxx = (img.at<NEWSIFT_wt>(r, c + 1) + img.at<NEWSIFT_wt>(r, c - 1) - v2)*second_deriv_scale;
			float dyy = (img.at<NEWSIFT_wt>(r + 1, c) + img.at<NEWSIFT_wt>(r - 1, c) - v2)*second_deriv_scale;
			float dxy = (img.at<NEWSIFT_wt>(r + 1, c + 1) - img.at<NEWSIFT_wt>(r + 1, c - 1) -
				img.at<NEWSIFT_wt>(r - 1, c + 1) + img.at<NEWSIFT_wt>(r - 1, c - 1)) * cross_deriv_scale;
			float tr = dxx + dyy;
			float det = dxx * dyy - dxy * dxy;

			if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
				return false;
		}

		kpt.pt.x = (c + xc) * (1 << octv);
		kpt.pt.y = (r + xr) * (1 << octv);
		kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
		kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
		kpt.response = std::abs(contr);

		return true;
	}


	//
	// Detects features at extrema in DoG scale space.  Bad features are discarded
	// based on contrast and ratio of principal curvatures.
	void NEWSIFT::findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
		vector<ColorKeypoint>& keypoints) const
	{
		int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
		int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * NEWSIFT_FIXPT_SCALE);
		const int COLOR_HIST_BUCKET_COUNT = 2;
		float colorHist[COLOR_HIST_BUCKET_COUNT * COLOR_HIST_BUCKET_COUNT * COLOR_HIST_BUCKET_COUNT];
		ColorKeypoint colorKeypoint;

		keypoints.clear();

		for (int o = 0; o < nOctaves; o++)
			for (int i = 1; i <= nOctaveLayers; i++)
			{
				int idx = o*(nOctaveLayers + 2) + i;
				const Mat& img = dog_pyr[idx];
				const Mat& prev = dog_pyr[idx - 1];
				const Mat& next = dog_pyr[idx + 1];
				int step = (int)img.step1();
				int rows = img.rows, cols = img.cols;

				for (int r = NEWSIFT_IMG_BORDER; r < rows - NEWSIFT_IMG_BORDER; r++)
				{
					const NEWSIFT_wt* currptr = img.ptr<NEWSIFT_wt>(r);
					const NEWSIFT_wt* prevptr = prev.ptr<NEWSIFT_wt>(r);
					const NEWSIFT_wt* nextptr = next.ptr<NEWSIFT_wt>(r);

					for (int c = NEWSIFT_IMG_BORDER; c < cols - NEWSIFT_IMG_BORDER; c++)
					{
						NEWSIFT_wt val = currptr[c];

						// find local extrema with pixel accuracy
						if (std::abs(val) > threshold &&
							((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
							val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
							val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
							val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
							val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
							val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
							val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
							val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
							val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
							(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
							val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
							val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
							val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
							val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
							val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
							val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
							val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
							val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
						{
							int r1 = r, c1 = c, layer = i;
							if (!adjustLocalExtrema(dog_pyr, colorKeypoint, o, layer, r1, c1,
								nOctaveLayers, (float)contrastThreshold,
								(float)edgeThreshold, (float)sigma))
								continue;
							float scl_octv = colorKeypoint.size*0.5f / (1 << o);
							float colorMax = calcColorHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
								Point(c1, r1),
								cvRound(NEWSIFT_ORI_RADIUS * scl_octv),
								NEWSIFT_ORI_SIG_FCTR * scl_octv,
								colorHist, COLOR_HIST_BUCKET_COUNT);
							float colorThreshold = (float)(colorMax * NEWSIFT_ORI_PEAK_RATIO);

							// Indices, not values
							int aroundI[2];
							int aroundJ[2];
							int aroundK[2];
							// Values, not indices, 0,1 are i values; 2,3 are j values; 4,5 are k values
							float surroundingVals[6];

							#define RED 0
							#define GREEN 1
							#define BLUE 2
							#define PREV 0
							#define NEXT 1
							#define PREV_I 0
							#define NEXT_I 1
							#define PREV_J 2
							#define NEXT_J 3
							#define PREV_K 4
							#define NEXT_K 5
							for (int i = 0; i < COLOR_HIST_BUCKET_COUNT; i++)
							{
								aroundI[PREV] = (i == 0) ? COLOR_HIST_BUCKET_COUNT - 1 : i - 1;
								aroundI[NEXT] = (i == COLOR_HIST_BUCKET_COUNT - 1) ? 0 : i + 1;
								for (int j = 0; j < COLOR_HIST_BUCKET_COUNT; j++)
								{
									aroundJ[PREV] = (j == 0) ? COLOR_HIST_BUCKET_COUNT - 1 : j - 1;
									aroundJ[NEXT] = (j == COLOR_HIST_BUCKET_COUNT - 1) ? 0 : j + 1;

									for (int k = 0; k < COLOR_HIST_BUCKET_COUNT; k++)
									{
										aroundK[PREV] = (k == 0) ? COLOR_HIST_BUCKET_COUNT - 1 : k - 1;
										aroundK[NEXT] = (k == COLOR_HIST_BUCKET_COUNT - 1) ? 0 : k + 1;

										int index = INDEX_3D_TO_1D(i, j, k, COLOR_HIST_BUCKET_COUNT);
										surroundingVals[PREV_I] = colorHist[INDEX_3D_TO_1D(aroundI[PREV], j, k, COLOR_HIST_BUCKET_COUNT)];
										surroundingVals[NEXT_I] = colorHist[INDEX_3D_TO_1D(aroundI[NEXT], j, k, COLOR_HIST_BUCKET_COUNT)];
										surroundingVals[PREV_J] = colorHist[INDEX_3D_TO_1D(i, aroundJ[PREV], k, COLOR_HIST_BUCKET_COUNT)];
										surroundingVals[NEXT_J] = colorHist[INDEX_3D_TO_1D(i, aroundJ[NEXT], k, COLOR_HIST_BUCKET_COUNT)];
										surroundingVals[PREV_K] = colorHist[INDEX_3D_TO_1D(i, j, aroundK[PREV], COLOR_HIST_BUCKET_COUNT)];
										surroundingVals[NEXT_K] = colorHist[INDEX_3D_TO_1D(i, j, aroundK[NEXT], COLOR_HIST_BUCKET_COUNT)];

										if ( // Compare previous and next x-axis values in the hist
											surroundingVals[PREV_I] < colorHist[index] &&
											surroundingVals[NEXT_I] < colorHist[index] &&
											// Compare previous and next y-axis values in the hist
											surroundingVals[PREV_J] < colorHist[index] &&
											surroundingVals[NEXT_J] < colorHist[index] &&
											// Compare previous and next z-axis values in the hist
											surroundingVals[PREV_K] < colorHist[index] &&
											surroundingVals[NEXT_K] < colorHist[index] &&
											colorHist[index] > colorThreshold)
										{
											float red = i + 0.5f *
												(surroundingVals[PREV_I] - surroundingVals[NEXT_I]) /
												(surroundingVals[PREV_I] - 2 * colorHist[index] + surroundingVals[NEXT_I]);
											red = (red < 0) ? red + COLOR_HIST_BUCKET_COUNT :
												(red > COLOR_HIST_BUCKET_COUNT) ? red - COLOR_HIST_BUCKET_COUNT : red;

											float green = i + 0.5f *
												(surroundingVals[PREV_J] - surroundingVals[NEXT_J]) /
												(surroundingVals[PREV_J] - 2 * colorHist[index] + surroundingVals[NEXT_J]);
											green = (green < 0) ? green + COLOR_HIST_BUCKET_COUNT :
												(green > COLOR_HIST_BUCKET_COUNT) ? green - COLOR_HIST_BUCKET_COUNT : green;

											float blue = i + 0.5f *
												(surroundingVals[PREV_K] - surroundingVals[NEXT_K]) /
												(surroundingVals[PREV_K] - 2 * colorHist[index] + surroundingVals[NEXT_K]);
											blue = (blue < 0) ? blue + COLOR_HIST_BUCKET_COUNT :
												(blue > COLOR_HIST_BUCKET_COUNT) ? blue - COLOR_HIST_BUCKET_COUNT : blue;

											colorKeypoint.color(RED) = (uchar) red;
											colorKeypoint.color(GREEN) = (uchar) green;
											colorKeypoint.color(BLUE) = (uchar) blue;

											keypoints.push_back(colorKeypoint);
										}
									}
								}
							}
						}
					}
				}
			}
	}
	void NEWSIFT::findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
		vector<KeyPoint>& keypoints) const
	{
		int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
		int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * NEWSIFT_FIXPT_SCALE);
		const int n = NEWSIFT_ORI_HIST_BINS;
		float hist[n];
		KeyPoint kpt;

		keypoints.clear();

		for (int o = 0; o < nOctaves; o++)
			for (int i = 1; i <= nOctaveLayers; i++)
			{
				int idx = o*(nOctaveLayers + 2) + i;
				const Mat& img = dog_pyr[idx];
				const Mat& prev = dog_pyr[idx - 1];
				const Mat& next = dog_pyr[idx + 1];
				int step = (int)img.step1();
				int rows = img.rows, cols = img.cols;

				for (int r = NEWSIFT_IMG_BORDER; r < rows - NEWSIFT_IMG_BORDER; r++)
				{
					const NEWSIFT_wt* currptr = img.ptr<NEWSIFT_wt>(r);
					const NEWSIFT_wt* prevptr = prev.ptr<NEWSIFT_wt>(r);
					const NEWSIFT_wt* nextptr = next.ptr<NEWSIFT_wt>(r);

					for (int c = NEWSIFT_IMG_BORDER; c < cols - NEWSIFT_IMG_BORDER; c++)
					{
						NEWSIFT_wt val = currptr[c];

						// find local extrema with pixel accuracy
						if (std::abs(val) > threshold &&
							((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
							val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
							val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
							val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
							val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
							val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
							val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
							val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
							val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
							(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
							val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
							val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
							val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
							val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
							val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
							val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
							val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
							val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
						{
							int r1 = r, c1 = c, layer = i;
							if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
								nOctaveLayers, (float)contrastThreshold,
								(float)edgeThreshold, (float)sigma))
								continue;
							float scl_octv = kpt.size*0.5f / (1 << o);
							float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer],
								Point(c1, r1),
								cvRound(NEWSIFT_ORI_RADIUS * scl_octv),
								NEWSIFT_ORI_SIG_FCTR * scl_octv,
								hist, n);
							float mag_thr = (float)(omax * NEWSIFT_ORI_PEAK_RATIO);
							for (int j = 0; j < n; j++)
							{
								int l = j > 0 ? j - 1 : n - 1;
								int r2 = j < n - 1 ? j + 1 : 0;

								if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
								{
									float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
									bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
									kpt.angle = 360.f - (float)((360.f / n) * bin);
									if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
										kpt.angle = 0.f;
									keypoints.push_back(kpt);
								}
							}
						}
					}
				}
			}
	}

	static void calcNEWSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst)
	{
		Point pt(cvRound(ptf.x), cvRound(ptf.y));
		float cos_t = cosf(ori*(float)(CV_PI / 180));
		float sin_t = sinf(ori*(float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = NEWSIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
		cos_t /= hist_width;
		sin_t /= hist_width;

		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
		int rows = img.rows, cols = img.cols;

		AutoBuffer<float> buf(len * 6 + histlen);
		float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
		float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

		for (i = 0; i < d + 2; i++)
		{
			for (j = 0; j < d + 2; j++)
				for (k = 0; k < n + 2; k++)
					hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
		}

		for (i = -radius, k = 0; i <= radius; i++)
			for (j = -radius; j <= radius; j++)
			{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				float dx = (float)(img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1));
				float dy = (float)(img.at<NEWSIFT_wt>(r - 1, c) - img.at<NEWSIFT_wt>(r + 1, c));
				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
				W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
				k++;
			}
			}

		len = k;
		fastAtan2(Y, X, Ori, len, true);
		magnitude(X, Y, Mag, len);
		exp(W, W, len);

		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin = (Ori[k] - ori)*bins_per_rad;
			float mag = Mag[k] * W[k];

			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);
			int o0 = cvFloor(obin);
			rbin -= r0;
			cbin -= c0;
			obin -= o0;

			if (o0 < 0)
				o0 += n;
			if (o0 >= n)
				o0 -= n;

			// histogram update using tri-linear interpolation
			float v_r1 = mag*rbin, v_r0 = mag - v_r1;
			float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
			float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
			float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
			float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
			float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

			int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
			hist[idx] += v_rco000;
			hist[idx + 1] += v_rco001;
			hist[idx + (n + 2)] += v_rco010;
			hist[idx + (n + 3)] += v_rco011;
			hist[idx + (d + 2)*(n + 2)] += v_rco100;
			hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
			hist[idx + (d + 3)*(n + 2)] += v_rco110;
			hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
			{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
				dst[(i*d + j)*n + k] = hist[idx + k];
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = d*d*n;
		for (k = 0; k < len; k++)
			nrm2 += dst[k] * dst[k];
		float thr = std::sqrt(nrm2)*NEWSIFT_DESCR_MAG_THR;
		for (i = 0, nrm2 = 0; i < k; i++)
		{
			float val = std::min(dst[i], thr);
			dst[i] = val;
			nrm2 += val*val;
		}
		nrm2 = NEWSIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
		for (k = 0; k < len; k++)
		{
			dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
		}
#else
		float nrm1 = 0;
		for (k = 0; k < len; k++)
		{
			dst[k] *= nrm2;
			nrm1 += dst[k];
		}
		nrm1 = 1.f / std::max(nrm1, FLT_EPSILON);
		for (k = 0; k < len; k++)
		{
			dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*NEWSIFT_INT_DESCR_FCTR);
		}
#endif
	}


	static void calcNEWSIFTDescriptor(const Mat& img, Point2f ptf, float ori, Vec3b color, float scl,
		int d, int n, float* dst)
	{
		Point pt(cvRound(ptf.x), cvRound(ptf.y));
		float cos_t = cosf(ori*(float)(CV_PI / 180));
		float sin_t = sinf(ori*(float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = NEWSIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
		cos_t /= hist_width;
		sin_t /= hist_width;

		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
		int rows = img.rows, cols = img.cols;

		AutoBuffer<float> buf(len * 6 + histlen);
		float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
		float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

		for (i = 0; i < d + 2; i++)
		{
			for (j = 0; j < d + 2; j++)
				for (k = 0; k < n + 2; k++)
					hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
		}

		for (i = -radius, k = 0; i <= radius; i++)
			for (j = -radius; j <= radius; j++)
			{
				// Calculate sample's histogram array coords rotated relative to ori.
				// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
				// r_rot = 1.5) have full weight placed in row 1 after interpolation.
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;
				float rbin = r_rot + d / 2 - 0.5f;
				float cbin = c_rot + d / 2 - 0.5f;
				int r = pt.y + i, c = pt.x + j;

				if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
					r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
				{
					float dx = (float)(img.at<NEWSIFT_wt>(r, c + 1) - img.at<NEWSIFT_wt>(r, c - 1));
					float dy = (float)(img.at<NEWSIFT_wt>(r - 1, c) - img.at<NEWSIFT_wt>(r + 1, c));
					X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
					k++;
				}
			}

		len = k;
		fastAtan2(Y, X, Ori, len, true);
		magnitude(X, Y, Mag, len);
		exp(W, W, len);

		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin = (Ori[k] - ori)*bins_per_rad;
			float mag = Mag[k] * W[k];

			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);
			int o0 = cvFloor(obin);
			rbin -= r0;
			cbin -= c0;
			obin -= o0;

			if (o0 < 0)
				o0 += n;
			if (o0 >= n)
				o0 -= n;

			// histogram update using tri-linear interpolation
			float v_r1 = mag*rbin, v_r0 = mag - v_r1;
			float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
			float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
			float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
			float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
			float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

			int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
			hist[idx] += v_rco000;
			hist[idx + 1] += v_rco001;
			hist[idx + (n + 2)] += v_rco010;
			hist[idx + (n + 3)] += v_rco011;
			hist[idx + (d + 2)*(n + 2)] += v_rco100;
			hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
			hist[idx + (d + 3)*(n + 2)] += v_rco110;
			hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
			{
				int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
				hist[idx] += hist[idx + n];
				hist[idx + 1] += hist[idx + n + 1];
				for (k = 0; k < n; k++)
					dst[(i*d + j)*n + k] = hist[idx + k];
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = d*d*n;
		for (k = 0; k < len; k++)
			nrm2 += dst[k] * dst[k];
		float thr = std::sqrt(nrm2)*NEWSIFT_DESCR_MAG_THR;
		for (i = 0, nrm2 = 0; i < k; i++)
		{
			float val = std::min(dst[i], thr);
			dst[i] = val;
			nrm2 += val*val;
		}
		nrm2 = NEWSIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
		for (k = 0; k < len; k++)
		{
			dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
		}
#else
		float nrm1 = 0;
		for (k = 0; k < len; k++)
		{
			dst[k] *= nrm2;
			nrm1 += dst[k];
		}
		nrm1 = 1.f / std::max(nrm1, FLT_EPSILON);
		for (k = 0; k < len; k++)
		{
			dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*NEWSIFT_INT_DESCR_FCTR);
		}
#endif
	}

	static void calcDescriptors(const vector<Mat>& gpyr, const vector<ColorKeypoint>& keypoints,
		Mat& descriptors, int nOctaveLayers, int firstOctave)
	{
		int d = NEWSIFT_DESCR_WIDTH, n = NEWSIFT_DESCR_HIST_BINS;

		for (size_t i = 0; i < keypoints.size(); i++)
		{
			ColorKeypoint kpt = keypoints[i];
			int octave, layer;
			float scale;
			unpackOctave(kpt, octave, layer, scale);
			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
			float size = kpt.size*scale;
			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
			const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];
			
			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;
			Vec3b color = kpt.color;
			calcNEWSIFTDescriptor(img, ptf, angle, color, size*0.5f, d, n, descriptors.ptr<float>((int)i));
		}
	}

	static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
		Mat& descriptors, int nOctaveLayers, int firstOctave)
	{
		int d = NEWSIFT_DESCR_WIDTH, n = NEWSIFT_DESCR_HIST_BINS;

		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale;
			unpackOctave(kpt, octave, layer, scale);
			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
			float size = kpt.size*scale;
			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
			const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;
			calcNEWSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////

	NEWSIFT::NEWSIFT(int _nfeatures, int _nOctaveLayers,
		double _contrastThreshold, double _edgeThreshold, double _sigma)
		: nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
		contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
	{
	}

	int NEWSIFT::descriptorSize() const
	{
		return NEWSIFT_DESCR_WIDTH*NEWSIFT_DESCR_WIDTH*NEWSIFT_DESCR_HIST_BINS;
	}

	int NEWSIFT::descriptorType() const
	{
		return CV_32F;
	}


	void NEWSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints) const
	{
		(*this)(_image, _mask, keypoints, noArray());
	}

	void NEWSIFT::operator()(InputArray _image, InputArray _mask,
		vector<ColorKeypoint>& keypoints,
		OutputArray _descriptors,
		bool useProvidedKeypoints) const
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
			ColorKeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				ColorKeyPointsFilter::retainBest(keypoints, nfeatures);
			//t = (double)getTickCount() - t;
			//printf("keypoint detection time: %g\n", t*1000./tf);

			if (firstOctave < 0)
				for (size_t i = 0; i < keypoints.size(); i++)
				{
					ColorKeypoint& kpt = keypoints[i];
					float scale = 1.f / (float)(1 << -firstOctave);
					kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
					kpt.pt *= scale;
					kpt.size *= scale;
				}

			if (!mask.empty())
				ColorKeyPointsFilter::runByPixelsMask(keypoints, mask);
		}
		else
		{
			// filter keypoints by mask
			//KeyPointsFilter::runByPixelsMask( keypoints, mask );
		}

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();

			calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}

	void NEWSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints,
		OutputArray _descriptors,
		bool useProvidedKeypoints) const
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
		else
		{
			// filter keypoints by mask
			//KeyPointsFilter::runByPixelsMask( keypoints, mask );
		}

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();

			calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}

	void NEWSIFT::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
	{
		(*this)(image, mask, keypoints, noArray());
	}

	void NEWSIFT::computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		(*this)(image, Mat(), keypoints, descriptors, true);
	}

}
