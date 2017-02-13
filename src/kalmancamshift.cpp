#include "main.hpp"
#include "LBPHue.hpp"
#include "kalmanFilter.hpp"

using namespace std;
using namespace cv;

static inline void adaptMyRect(Rect A,Rect& B){
    if(B.x < 0)
        B.x = 0;
    if(B.x+B.width>A.width)
        B.width = A.width - B.x;
    if(B.y < 0)
        B.y = 0;
    if(B.y+B.height>A.height)
        B.height = A.height - B.y;
}

/************************************************/
int select_object = 0, start_track = 0, track_object = 0;

Point2f origin;
Rect selection, origin_box;



static void on_mouse( int event, int x, int y, int flags ,void *p) {
    if( select_object ) {
        selection.x = MIN(x,origin.x);
        selection.y = MIN(y,origin.y);
        selection.width = selection.x + CV_IABS(x - origin.x);
        selection.height = selection.y + CV_IABS(y - origin.y);

        selection.x = MAX( selection.x, 0 );
        selection.y = MAX( selection.y, 0 );
        selection.width = selection.width;
        selection.height = selection.height;
        selection.width -= selection.x;
        selection.height -= selection.y;

    }

    switch( event ) {
    case CV_EVENT_LBUTTONDOWN:
        origin = cvPoint(x,y);
        selection = cvRect(x,y,0,0);
        select_object = 1;
        break;
    case CV_EVENT_LBUTTONUP:
        select_object = 0;
        if( selection.width > 0 && selection.height > 0 )
            track_object = -1;
        origin_box=selection;

        printf("n # 鼠标的选择区域：");
        printf("n X = %d, Y = %d, Width = %d, Height = %d",
            selection.x, selection.y, selection.width, selection.height);
        break;
    }
}
/*************************************main***************************/
int kalmen_camshift(VideoCapture capture, Mat background = Mat()){
	Rect search_window, track_window;
	Mat frame, image, hsv,mask;
	namedWindow( "CamShiftDemo", 1 );
    setMouseCallback( "CamShiftDemo", on_mouse, NULL ); // on_mouse 自定义事件
    MyKalmanFilter posKal = MyKalmanFilter();
    MyKalmanFilter areaKal = MyKalmanFilter();
    LBPMixHist ra = LBPMixHist();

    while(capture.read(frame))
    {
    	clock_t t1 = clock();
    	int videoWidth = frame.size().width / 8;
    	int videoHeight = frame.size().height / 8;

		GaussianBlur(frame, image,Size(7,7),1.6,1.6);
		image = frame + 3 * (frame - image) - background; //高反差保留
//		cout<<temp.channels()<<endl;

		cvtColor(image, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(0,10,50,0),Scalar(180,255,255,0), mask);

        if(track_object) {
            if(track_object < 0){
                ra.setRoiHist(hsv(origin_box));
                track_window = selection;
                track_object=1;
                posKal.firstInit(selection);
                areaKal.firstInit(selection);
                namedWindow("CamShiftDemo1",0);
            }

            Point perdict_center = posKal.MykalmanPointPredict();
            Rect perdict_window = areaKal.MykalmanRectPredict();

            track_window = Rect(perdict_center.x - 0.5 * perdict_window.width,
            					perdict_center.y - 0.5 * perdict_window.height,
            					perdict_window.width, perdict_window.height );

            adaptMyRect(Rect(0,0,frame.size().width,frame.size().height),track_window);
			//只对目标周围计算投影
			search_window = Rect(track_window.x - videoWidth,
								 track_window.y - videoHeight,
								 track_window.width + videoWidth * 2,
								 track_window.height + videoHeight * 2 );
			//修正search_window，使其范围在图像内
			adaptMyRect(Rect(0,0,frame.size().width,frame.size().height),search_window);
			Mat temp = ra.getBackprojImage(hsv(search_window),mask(search_window));
            //因为设置了_1ROI，所以更新track_window
            Rect _track_window = Rect(0.5 * perdict_window.width, 0.5 * perdict_window.height,
            						  perdict_window.width, perdict_window.height);
            // calling CAMSHIFT 算法模块
            RotatedRect track_box = CamShift( temp, _track_window,
            		cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 0.5 ));
            track_window = Rect(search_window.x + track_box.boundingRect().x,
            					search_window.y + track_box.boundingRect().y,
            					track_box.boundingRect().width,
            					track_box.boundingRect().height);
            ellipse( temp, track_box, Scalar(0,0,255), 3, CV_AA );imshow( "CamShiftDemo1", temp );
            rectangle(image ,Point(track_window.x,track_window.y),
                             Point(track_window.x+track_window.width,track_window.y+track_window.height),
                             CV_RGB(255,0,0),2, 8, 0);

            posKal.kalmanCorrect(track_box.center);
            areaKal.kalmanCorrect(track_box.boundingRect());

        }

        if( select_object && selection.width > 0 && selection.height > 0 ) {
        	Mat select_roi(image, selection);
        	bitwise_not(select_roi, select_roi);
        }

        clock_t t2 = clock();
        imshow( "CamShiftDemo", image );
        waitKey(1);
        cout << (double)(t2 - t1) / CLOCKS_PER_SEC * 1000 << endl;
    }
    return 0;
}
