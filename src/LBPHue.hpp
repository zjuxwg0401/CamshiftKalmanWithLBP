#include "main.hpp"

using namespace std;
using namespace cv;


#ifndef _NLBP_
#define _NLBP_

class NLBP{
private:
	static const uchar lbp_data[256];

	static inline double getK(double i){
		if(i > 0 && i <= 1) return -0.2/(1+exp(1-i));
		else if(i > 1 && i < 2) return -0.1/(1+exp(1-i));
		else return 0.1/(1+exp(1-i));
	}

public:

	Mat GRAY2UniformNLBP(const Mat &image){
		Mat result(Size(image.cols, image.rows), image.type());
		uchar neighbor[8] = {0};
		uchar temp;
		CV_Assert( image.channels() == 1 );
		for(int y = 1; y < image.rows-1; ++y)
		{
			for(int x = 1; x < image.cols-1; ++x){
				neighbor[0] = image.at<uchar>(y - 1, x - 1);
				neighbor[1] = image.at<uchar>(y - 1, x);
				neighbor[2] = image.at<uchar>(y - 1, x + 1);
				neighbor[3] = image.at<uchar>(y, x + 1);
				neighbor[4] = image.at<uchar>(y + 1, x + 1);
				neighbor[5] = image.at<uchar>(y + 1, x);
				neighbor[6] = image.at<uchar>(y + 1, x - 1);
				neighbor[7] = image.at<uchar>(y , x - 1);
				vector<int> temp_im(neighbor, neighbor+8);
				Mat temp_m, temp_sd;
				meanStdDev(temp_im,temp_m,temp_sd);
				double m = temp_m.at<double>(0,0);
				double sd = temp_sd.at<double>(0,0);

				int t = (int)getK(m/(sd*sd)) * (int)sd + (int)m;
				temp = 0;
				for(int k = 0; k < 8; ++k) {
					temp += (neighbor[k] >= t) << k;
				}

				result.at<uchar>(y,x) = lbp_data[temp];
			}
		}
		return result;
	}

	void GRAY2NormalUniformLBP(Mat image, Mat &dst){
		uchar neighbor[8] = {0};
		uchar temp;
		CV_Assert( image.channels() == 1 && dst.channels() == 1 );
		for(int y = 1; y < image.rows-1; ++y)
		{
			for(int x = 1; x < image.cols-1; ++x){
				neighbor[0] = image.at<uchar>(y - 1, x - 1);
				neighbor[1] = image.at<uchar>(y - 1, x);
				neighbor[2] = image.at<uchar>(y - 1, x + 1);
				neighbor[3] = image.at<uchar>(y, x + 1);
				neighbor[4] = image.at<uchar>(y + 1, x + 1);
				neighbor[5] = image.at<uchar>(y + 1, x);
				neighbor[6] = image.at<uchar>(y + 1, x - 1);
				neighbor[7] = image.at<uchar>(y , x - 1);
				uchar center = image.at<uchar>(y, x);
				temp = 0;
				for(int k = 0; k < 8; ++k) {
					temp += (neighbor[k] >= center) << k;
				}

				dst.at<uchar>(y, x) = lbp_data[temp];

			}
		}
	}

	Mat GRAY2NormalUniformLBP(const Mat &image){
		uchar neighbor[8] = {0};
		uchar temp;
		Mat result(image.rows, image.cols, image.type(), Scalar(1));
		CV_Assert( image.channels() == 1 );
		for(int y = 1; y < image.rows-1; ++y) {
			for(int x = 1; x < image.cols-1; ++x){
				neighbor[0] = image.at<uchar>(y - 1, x - 1);
				neighbor[1] = image.at<uchar>(y - 1, x);
				neighbor[2] = image.at<uchar>(y - 1, x + 1);
				neighbor[3] = image.at<uchar>(y, x + 1);
				neighbor[4] = image.at<uchar>(y + 1, x + 1);
				neighbor[5] = image.at<uchar>(y + 1, x);
				neighbor[6] = image.at<uchar>(y + 1, x - 1);
				neighbor[7] = image.at<uchar>(y , x - 1);
				uchar center = image.at<uchar>(y, x);
				temp = 0;
				for(int k = 0; k < 8; ++k) {
					temp += (neighbor[k] >= center) << k;
				}

				result.at<uchar>(y, x) = lbp_data[temp];

			}
		}
		return result;
	}



};
const uchar NLBP::lbp_data[256] = {
		0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0, 2, 3,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 3, 4,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 5,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 5, 0,
		0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 2, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		2, 3, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0,
		3, 4, 0, 5, 0, 0, 0, 0, 4, 5, 0, 0, 5, 0, 0, 0
};

#endif


#ifndef _LBPMixHist_
#define _LBPMixHist_

class LBPMixHist : public NLBP{
	private:
			int HistSize[2];
			float LBPrange[2];
			float Hueranges[2];
			const float* phranges[2];
			int img_cls[2];

			Mat mix_img,backproj;
			MatND roi_hist;

			Mat temp_img,temp_mask;

	public:
			LBPMixHist(const Mat& image, const Mat& mask, float hueHistSize = 16,
					float hueMin = 0., float hueMax = 180.)
			: temp_img(image) ,temp_mask(mask){
				CV_Assert( image.channels() == 3 && image.elemSize1() == 1);
				HistSize[0] = hueHistSize;
				HistSize[1] = 6;
				LBPrange[1] = 5.;
				LBPrange[0] = 0.;
				Hueranges[0] = hueMin; Hueranges[1] = hueMax;
				img_cls[0] = 0; img_cls[1] = 2;
				phranges[0] = Hueranges;
				phranges[1] = LBPrange;
			}
			LBPMixHist(float hueMin = 0.f){
				HistSize[0] = 16;
				HistSize[1] = 6;
				LBPrange[1] = 5.;
				LBPrange[0] = 0.;
				Hueranges[0] = hueMin; Hueranges[1] = 180.f;
				img_cls[0] = 0; img_cls[1] = 2;
				phranges[0] = Hueranges;
				phranges[1] = LBPrange;
			}

			void getBackprojImage(Mat image, Rect RoiRect, Mat mask = Mat()){
				CV_Assert( image.channels() == 3 && mask.elemSize1() == 1);
				temp_img = image(RoiRect);
				vector<Mat> cls(3);;
				split(temp_img,cls);
				cls[2] = GRAY2NormalUniformLBP(cls[2]);
				inRange(cls[1], 10, 255, temp_mask);
				merge(cls,mix_img);
				img_cls[0] = 0; img_cls[1] = 2;
				namedWindow("t1", 0);imshow("t1", temp_mask);
				if(!mask.empty())
					temp_mask &= mask(RoiRect);
				namedWindow("t2", 0);imshow("t2", temp_mask);
				calcHist(&mix_img, 1, img_cls, temp_mask, roi_hist, 2, HistSize, phranges);
				normalize(roi_hist,roi_hist,0,255,CV_MINMAX);
				temp_img = image;
				split(temp_img,cls);
				cls[2] = GRAY2NormalUniformLBP(cls[2]);
				merge(cls,mix_img);
				calcBackProject(&mix_img, 1, img_cls, roi_hist, backproj, phranges);
				medianBlur(backproj, backproj, 7);
			}

			void setRoiHist(Mat image, Mat mask = Mat()){
				CV_Assert( image.channels() == 3 && mask.elemSize1() == 1);
				temp_img = image;temp_mask = mask;
				vector<Mat> cls(3);
				split(temp_img,cls);
				cls[2] = GRAY2NormalUniformLBP(cls[2]);
				inRange(cls[1], 50, 255, temp_mask);
				merge(cls,mix_img);
				img_cls[0] = 0; img_cls[1] = 2;
				if(!mask.empty())
					temp_mask &= mask;
				calcHist(&mix_img, 1, img_cls, temp_mask, roi_hist, 2, HistSize, phranges);
				normalize(roi_hist,roi_hist,0,255,CV_MINMAX);
			}

			void setRoiHist(Mat image, Mat mask, Rect rect){
				CV_Assert( image.channels() == 3 && mask.elemSize1() == 1);
				temp_img = image(rect);
				temp_mask = mask(rect);
				vector<Mat> cls(3);
				split(temp_img,cls);
				cls[2] = GRAY2NormalUniformLBP(cls[2]);
				inRange(cls[1], 50, 255, temp_mask);
				merge(cls,mix_img);
				img_cls[0] = 0; img_cls[1] = 2;
				if(!mask.empty())
					temp_mask &= mask(rect);
				calcHist(&mix_img, 1, img_cls, temp_mask, roi_hist, 2, HistSize, phranges);
//				normalize(roi_hist,roi_hist,0,255,CV_MINMAX);
			}

			Mat getBackprojImage(Mat image, Mat mask = Mat()){
				CV_Assert( image.channels() == 3 );

				temp_img = image;
				vector<Mat> cls(3);
				split(temp_img,cls);
				if(!mask.empty())
					cls[0] &= mask;
				cls[2] = GRAY2NormalUniformLBP(cls[2]);
				merge(cls,mix_img);
				calcBackProject(&mix_img, 1, img_cls, roi_hist, backproj, phranges);
				threshold(backproj,backproj, 205, 0, CV_THRESH_TOZERO);
				morphologyEx(backproj, backproj, MORPH_BLACKHAT, Mat(5, 5, CV_8U, Scalar(1)));
				return backproj;
			}

			Mat getRoiImg(){
				return temp_img;
			}

			Mat getMixRoiImg(){
				return mix_img;
			}

			Mat getRoiHist(){
				return roi_hist;
			}

			Mat getBackprojImage(){
				return backproj;
			}
};

#endif

