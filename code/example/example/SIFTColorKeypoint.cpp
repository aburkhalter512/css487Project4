#include "SIFTColorKeypoint.h"
namespace cv
{
	SIFTColorKeypoint::~SIFTColorKeypoint()
	{
	}

   void SIFTColorKeypoint::retainBest(vector<SIFTColorKeypoint>& keypoints, int n_points)
   {
      //this is only necessary if the keypoints size is greater than the number of desired points.
      if( n_points >= 0 && keypoints.size() > (size_t)n_points )
      {
         if (n_points==0)
         {
            keypoints.clear();
            return;
         }
         //first use nth element to partition the keypoints into the best and worst.
         std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
         //this is the boundary response, and in the case of FAST may be ambigous
         float ambiguous_response = keypoints[n_points - 1].response;
         //use std::partition to grab all of the keypoints with the boundary response.
         vector<KeyPoint>::const_iterator new_end =
               std::partition(keypoints.begin() + n_points, keypoints.end(),
               KeypointResponseGreaterThanThreshold(ambiguous_response));
         //resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
         keypoints.resize(new_end - keypoints.begin());
      }
   }

   void SIFTColorKeypoint::runByPixelsMask( vector<SIFTColorKeypoint>& keypoints, const Mat& mask )
   {
       if( mask.empty() )
           return;

       keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), MaskPredicate(mask)), keypoints.end());
   }

   void removeDuplicated( vector<SIFTColorKeypoint>& keypoints )
   {
       int i, j, n = (int)keypoints.size();
       vector<int> kpidx(n);
       vector<uchar> mask(n, (uchar)1);

       for( i = 0; i < n; i++ )
           kpidx[i] = i;
       std::sort(kpidx.begin(), kpidx.end(), KeyPoint_LessThan(keypoints));
       for( i = 1, j = 0; i < n; i++ )
       {
           KeyPoint& kp1 = keypoints[kpidx[i]];
           KeyPoint& kp2 = keypoints[kpidx[j]];
           if( kp1.pt.x != kp2.pt.x || kp1.pt.y != kp2.pt.y ||
               kp1.size != kp2.size || kp1.angle != kp2.angle )
               j = i;
           else
               mask[kpidx[i]] = 0;
       }

       for( i = j = 0; i < n; i++ )
       {
           if( mask[i] )
           {
               if( i != j )
                   keypoints[j] = keypoints[i];
               j++;
           }
       }
       keypoints.resize(j);
   }
}
