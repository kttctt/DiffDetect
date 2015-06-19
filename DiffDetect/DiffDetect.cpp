
#include <opencv2/opencv.hpp>
#include "../../../util/opencv_lib.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const float inlier_threshold = 0.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.5f;   // Nearest neighbor matching ratio

const int gap = 10;
//const int ransac_r = 5;
//const int ransac_n = 15;

//---------------------------------------------------------------
//【関数名　】：Test
//【処理概要】：
//【引数　　】：src        = 入力画像（三色8bit3ch）
//　　　　　　：dst        = 出力画像（三色8bit3ch）
//　　　　　　：
//　　　　　　：
//【戻り値　】：
//【備考　　】：
//--------------------------------------------------------------- 

void Test(const cv::Mat src, cv::Mat dst){

	cv::Mat dst_img(src.size(), CV_8U);
	cv::Mat gray_img(src.size(), CV_8U);
	cv::Mat bin_img[3];
	cv::Mat bgr_img[3];

	int src_channel = src.channels();	// channel 数

	if (src_channel == 1) bgr_img[0] = src.clone();	// 単色
	else cv::split(src, bgr_img);					// 三色(BGR)

	for (size_t i = 0; i < src_channel; i++){		// 処理ループ

		/*   何らかの処理      */


	} // 処理ループ

	if (src_channel == 1)  bgr_img[0].copyTo(dst);	// 単色
	else cv::merge(bgr_img, 3, dst);				// 三色(BGR)

	return;
}

int main(int argc, char **argv)
{
	//const int WIDTH = 640;  // 幅
	//const int HEIGHT = 480; // 高さ
	//const int CAMERANUM = 0; // カメラ番号

	//cv::VideoCapture capture(CAMERANUM); // デフォルトカメラをオープン
	//if (!capture.isOpened())  return -1;// 成功したかどうかをチェック

	//Mat frame;
	//Mat gray_img(HEIGHT, WIDTH, HEIGHT, CV_8UC1); // WIFTH*HEIGHTの単色
	//Mat bin_img(HEIGHT, WIDTH, CV_8UC1);
	//Mat result_img(HEIGHT, WIDTH, CV_8UC3);

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	Mat img1 = imread("../picture/cheeseL.jpg", 1);
	Mat img2 = imread("../picture/cheeseR.jpg", 1);
	Mat gray1, gray2;
	cvtColor(img1, gray1, cv::COLOR_RGB2GRAY);
	cvtColor(img2, gray2, cv::COLOR_RGB2GRAY);
	Mat combined_img = Mat::zeros(Size(img1.size().width + img2.size().width + gap, max(img1.size().height, img2.size().height)), CV_8UC3);
	Rect rect1(0, 0, img1.size().width, img1.size().height);
	Rect rect2(img1.size().width + gap, 0, img2.size().width, img2.size().height);
	Mat roi1(combined_img, rect1);
	Mat roi2(combined_img, rect2);
	//roi1 = img1.clone();
	//roi2 = img2.clone();
	img1.copyTo(roi1);
	img2.copyTo(roi2);
	//result_img = img2.clone();
	Mat old_gray, and_gray;
	old_gray = Mat::zeros(img1.size(),CV_8UC1);

	float angle = -40.0, scale = 1.0;
	cv::Point2f center(img2.cols*0.5, img2.rows*0.5);
	cv::Mat img2_rotate, gray2_rotate;
	cv::Mat affine_matrix;

	cv::Point2f pts1[] = { cv::Point2f(150, 150), cv::Point2f(150, 300), cv::Point2f(350, 300), cv::Point2f(350, 150) };
	cv::Point2f pts2[] = { cv::Point2f(170, 150), cv::Point2f(170, 300), cv::Point2f(340, 290), cv::Point2f(340, 160) };
	cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);

	affine_matrix = cv::getRotationMatrix2D(center, angle, scale);
	cv::warpAffine(img2, img2_rotate, affine_matrix, img2.size());
	cv::warpAffine(gray2, gray2_rotate, affine_matrix, img2.size());
	cv::warpPerspective(img2_rotate, img2_rotate, perspective_matrix, img2_rotate.size(), cv::INTER_LINEAR);
	cv::warpPerspective(gray2_rotate, gray2_rotate, perspective_matrix, gray2_rotate.size(), cv::INTER_LINEAR);
	img2_rotate.copyTo(roi2);


	Ptr<AKAZE> akaze = AKAZE::create();
	//akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(gray1, noArray(), kpts1, desc1);

	//cv::drawKeypoints(img1, kpts1, img1);
	imshow("input", img1); // 結果表示


	while (1) {

		combined_img = Mat::zeros(Size(img1.size().width + img2.size().width + gap, max(img1.size().height, img2.size().height)), CV_8UC3);
		angle += 1.0f;
		scale = 0.8f + 0.3f * sin(angle/180.0f*CV_PI);
		affine_matrix = cv::getRotationMatrix2D(center, angle, scale);
		cv::warpAffine(img2, img2_rotate, affine_matrix, img2.size());
		cv::warpAffine(gray2, gray2_rotate, affine_matrix, img2.size());
		//cv::warpPerspective(img2_rotate, img2_rotate, perspective_matrix, img2_rotate.size(), cv::INTER_LINEAR);
		//cv::warpPerspective(gray2_rotate, gray2_rotate, perspective_matrix, gray2_rotate.size(), cv::INTER_LINEAR);
		img1.copyTo(roi1);
		img2_rotate.copyTo(roi2);

		//capture >> frame; // カメラから新しいフレームを取得

		//cvtColor(frame, gray_img, COLOR_BGR2GRAY);

		//result_img = frame.clone();

		akaze->detectAndCompute(gray2_rotate, noArray(), kpts2, desc2);
		BFMatcher matcher(NORM_HAMMING);
		vector< vector<DMatch> > nn_matches;
		matcher.knnMatch(desc1, desc2, nn_matches, 2);

		vector<KeyPoint> matched1, matched2, inliers1, inliers2;
		vector<DMatch> good_matches;
		vector<pair<KeyPoint, KeyPoint>> matched_pair;
		vector<Point2f> matched_pt1, matched_pt2, perspect_pt;
		for (size_t i = 0; i < nn_matches.size(); i++) {
			DMatch first = nn_matches[i][0];
			float dist1 = nn_matches[i][0].distance;
			float dist2 = nn_matches[i][1].distance;

			if (dist1 < nn_match_ratio * dist2) {
				matched1.push_back(kpts1[first.queryIdx]);
				matched2.push_back(kpts2[first.trainIdx]);
				matched_pair.push_back(pair<KeyPoint, KeyPoint>(kpts1[first.queryIdx], kpts2[first.trainIdx]));
				matched_pt1.push_back(kpts1[first.queryIdx].pt);
				matched_pt2.push_back(kpts2[first.trainIdx].pt);
			}
		}

		//cv::drawKeypoints(result_img, matched1, result_img);

		for (auto k : matched_pair){
			line(combined_img, k.first.pt, k.second.pt+Point2f(img1.size().width+gap,0), Scalar(125));

		}

		//imshow("result", result_img); // 結果表示
		imshow("combine", combined_img);
		//imshow("capture", frame); // カメラ画像表示

		if (matched1.size() < 4 || matched2.size() < 4) continue;

		Mat H = findHomography(matched_pt2, matched_pt1, CV_RANSAC);
		perspectiveTransform(matched_pt2, perspect_pt, H);
		vector<double> norms;
		for (int i = 0; i < perspect_pt.size(); ++i){
			//cout << norm(matched_pt1[i] - perspect_pt[i]) << endl;
			norms.push_back(norm(matched_pt1[i] - perspect_pt[i]));
		}
		std::sort(norms.begin(), norms.end());


		Mat img2_out;
		cv::warpPerspective(img2_rotate, img2_out, H, img2_rotate.size());
		Mat img2_out_resize = img2_out(rect1);
		imshow("out", img2_out);
		Mat diff;
		absdiff(img1, img2_out_resize, diff);
		//cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), 1);
		//cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1), 4);
		//cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), 3);
		imshow("diff", diff);

		Mat diff_gray;
		cvtColor(diff, diff_gray, cv::COLOR_RGB2GRAY);
		//adaptiveThreshold(diff_gray, diff_gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 1);

		GaussianBlur(diff_gray, diff_gray, Size(7, 7), 1.5);
		Canny(diff_gray, diff_gray, 10, 50);
		//threshold(diff_gray, diff_gray, 10, 255, cv::THRESH_BINARY);
		imshow("diff_gray", diff_gray);

		bitwise_and(diff_gray, old_gray, and_gray);
		old_gray = diff_gray.clone();
		imshow("and_gray", and_gray);


		int key = cv::waitKey(1);
		if (key == 'q') break; // キー入力で終了
		else if (key == 's'){


			//img1 = frame.clone();
			//akaze->detectAndCompute(img1, noArray(), kpts2, desc2);

			//cv::drawKeypoints(img1, kpts2, img1);
			//imshow("input", img1); // 結果表示

		}

	}
	return 0;
}

