#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

#include "main.hpp"

class MyKalmanFilter {
private:

	CvKalman* kalman;
	CvMat* measurement, *realposition;
	Point2f lastpoint;
	Rect lastRect;
	const CvMat* prediction;
	static const float A[16];

public:

	MyKalmanFilter():kalman(0), measurement(0), realposition(0), prediction(0){
		kalman = cvCreateKalman( 4, 2, 0 );
		memcpy( kalman->transition_matrix->data.fl, A, sizeof(A));//A
		cvSetIdentity( kalman->measurement_matrix, cvScalarAll(1) );//H
		cvSetIdentity( kalman->process_noise_cov, cvScalarAll(1e-5) );//Q w
		cvSetIdentity( kalman->measurement_noise_cov, cvScalarAll(1e-1) );//R v
		cvSetIdentity( kalman->error_cov_post, cvScalarAll(1));//P
		measurement = cvCreateMat( 2, 1, CV_32FC1 );//Z(k)
		realposition = cvCreateMat( 4, 1, CV_32FC1 );//real X(k)
	}

	~MyKalmanFilter() {
	   cvReleaseKalman(&kalman);
	}

	void kalmanCorrect(Rect window){
		realposition->data.fl[0] = window.width;
		realposition->data.fl[1] = window.height ;
		realposition->data.fl[2] = window.width - lastRect.width;
		realposition->data.fl[3] = window.height - lastRect.height;
		lastRect = window;//keep the current real position
		cvMatMulAdd( kalman->measurement_matrix/*2x4*/, realposition/*4x1*/,
		/*measurementstate*/ 0, measurement );
		cvKalmanCorrect( kalman, measurement );
	}

	void kalmanCorrect(Point2f a){
			realposition->data.fl[0] = a.x;
			realposition->data.fl[1] = a.y;
			realposition->data.fl[2] = a.x - lastpoint.x;
			realposition->data.fl[3] = a.y - lastpoint.y;
			lastpoint = a;//keep the current real position
			cvMatMulAdd( kalman->measurement_matrix/*2x4*/, realposition/*4x1*/,
			/*measurementstate*/ 0, measurement );
			cvKalmanCorrect( kalman, measurement );
	}

	void firstInit(Rect ra){
		lastRect = ra;
		float input[4] = {(float)(ra.width), (float)(ra.height), 0., 0.};
		memcpy( kalman->state_post->data.fl, input, sizeof(input));
	}

	void firstInit(Point2f ra){
		lastpoint = ra;
		float input[4] = {(float)ra.x, (float)ra.y, 0., 0.};
		memcpy( kalman->state_post->data.fl, input, sizeof(input));
	}

	Point MykalmanPointPredict(){
		prediction = cvKalmanPredict( kalman, 0 );//predicton=kalman->state_post
		return Point( cvRound(prediction->data.fl[0]), cvRound(prediction->data.fl[1]));
	}

	Rect MykalmanRectPredict(){
		prediction = cvKalmanPredict( kalman, 0 );//predicton=kalman->state_post
		return Rect( 0, 0, cvRound(prediction->data.fl[0]), cvRound(prediction->data.fl[1]));
	}

};

const float MyKalmanFilter::A[16] = {   1,0,1,0,
										0,1,0,1,
										0,0,1,0,
										0,0,0,1  };

#endif /* KALMANFILTER_H_ */
