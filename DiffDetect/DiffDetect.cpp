
#include <opencv2/opencv.hpp>
#include "../../../util/opencv_lib.hpp"
#include <iostream>
#include <vector>
#include <numeric>

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

void Labeling(const cv::Mat src, cv::Mat dst, int thresh=0){

	cv::Mat dst_tmp = cv::Mat::zeros(src.size(), CV_8U);

	//int src_channel = src.channels();	// channel 数
	//if (src_channel == 1) bgr_img[0] = src.clone();	// 単色
	//else cv::split(src, bgr_img);					// 三色(BGR)

	int min_id_default = 9999;
	int min_id = min_id_default;

	vector<int> index_hash;
	vector<int> index_sort(4);
	for (int y = 1; y < src.rows; ++y){
		for (int x = 1; x < src.cols; ++x){
			if (src.channels() == 3){
				if (src.data[y * src.step + x * src.elemSize() + 0] > thresh || src.data[y * src.step + x * src.elemSize() + 1] > thresh || src.data[y * src.step + x * src.elemSize() + 2] > thresh){
					index_sort[0] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					index_sort[1] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 0)* dst_tmp.elemSize()]);
					index_sort[2] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x + 1)* dst_tmp.elemSize()]);
					index_sort[3] = static_cast<int>(dst_tmp.data[(y - 0) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					sort(index_sort.begin(), index_sort.end());
					if (accumulate(index_sort.begin(), index_sort.end(), 0) == 0){
						index_hash.push_back(index_hash.size() + 1);
						dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = index_hash.size();
					}
					else{
						//cout << accumulate(index_sort.begin(), index_sort.end(), 0) << endl;
						min_id = min_id_default;
						for (int i = 0; i < 4; ++i){
							if (index_sort[i] == 0)continue;
							if (min_id == min_id_default){
								min_id = index_sort[i];
							}
							else{
								index_hash[index_sort[i]-1] = min_id;
								dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = min_id;
							}
						}
					}

					//cout << min_id << endl;

				}
			}
			else {
				if (src.data[y * src.step + x * src.elemSize()] > thresh){
					index_sort[0] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					index_sort[1] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 0)* dst_tmp.elemSize()]);
					index_sort[2] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x + 1)* dst_tmp.elemSize()]);
					index_sort[3] = static_cast<int>(dst_tmp.data[(y - 0) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					sort(index_sort.begin(), index_sort.end());
					if (accumulate(index_sort.begin(), index_sort.end(), 0) == 0){
						index_hash.push_back(index_hash.size() + 1);
						dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = index_hash.size();
					}
					else{
						//cout << accumulate(index_sort.begin(), index_sort.end(), 0) << endl;
						min_id = min_id_default;
						for (int i = 0; i < 4; ++i){
							if (index_sort[i] == 0)continue;
							if (min_id == min_id_default){
								min_id = index_sort[i];
							}
							else{
								index_hash[index_sort[i] - 1] = min_id;
								dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = min_id;
							}
						}
					}
				}
			}
		}
	}

	//if (src_channel == 1)  bgr_img[0].copyTo(dst);	// 単色
	//else cv::merge(bgr_img, 3, dst);				// 三色(BGR)
	std::sort(index_hash.begin(), index_hash.end());
	index_hash.erase(std::unique(index_hash.begin(), index_hash.end()), index_hash.end());

	for (auto v : index_hash){
		cout << v << endl;
	}
	threshold(dst_tmp, dst_tmp, 0, 255, cv::THRESH_BINARY);
	imshow("aaa", dst_tmp);
	waitKey(0);
	return;
}

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

void LabelingArea(const cv::Mat src, cv::Mat dst, int thresh = 0){

	cv::Mat dst_tmp = cv::Mat::zeros(src.size(), CV_8U);

	//int src_channel = src.channels();	// channel 数
	//if (src_channel == 1) bgr_img[0] = src.clone();	// 単色
	//else cv::split(src, bgr_img);					// 三色(BGR)

	int min_id_default = 9999;
	int min_id = min_id_default;

	vector<pair<int, int>> index_hash;
	vector<int> index_sort(4);
	for (int y = 1; y < src.rows; ++y){
		for (int x = 1; x < src.cols; ++x){
			if (src.channels() == 3){
				if (src.data[y * src.step + x * src.elemSize() + 0] > thresh || src.data[y * src.step + x * src.elemSize() + 1] > thresh || src.data[y * src.step + x * src.elemSize() + 2] > thresh){
					index_sort[0] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					index_sort[1] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 0)* dst_tmp.elemSize()]);
					index_sort[2] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x + 1)* dst_tmp.elemSize()]);
					index_sort[3] = static_cast<int>(dst_tmp.data[(y - 0) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					sort(index_sort.begin(), index_sort.end());
					if (accumulate(index_sort.begin(), index_sort.end(), 0) == 0){
						index_hash.push_back(pair<int,int>(index_hash.size() + 1,1));
						dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = index_hash.size();
					}
					else{
						//cout << accumulate(index_sort.begin(), index_sort.end(), 0) << endl;
						min_id = min_id_default;
						for (int i = 0; i < 4; ++i){
							if (index_sort[i] == 0)continue;
							if (min_id == min_id_default){
								min_id = index_sort[i];
							}
							else{
								index_hash[index_sort[i] - 1].first = min_id;
								index_hash[index_sort[i] - 1].second++;
								dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = min_id;
							}
						}
					}

					//cout << min_id << endl;

				}
			}
			else {
				if (src.data[y * src.step + x * src.elemSize()] > thresh){
					index_sort[0] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					index_sort[1] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x - 0)* dst_tmp.elemSize()]);
					index_sort[2] = static_cast<int>(dst_tmp.data[(y - 1) * dst_tmp.step + (x + 1)* dst_tmp.elemSize()]);
					index_sort[3] = static_cast<int>(dst_tmp.data[(y - 0) * dst_tmp.step + (x - 1)* dst_tmp.elemSize()]);
					sort(index_sort.begin(), index_sort.end());
					if (accumulate(index_sort.begin(), index_sort.end(), 0) == 0){
						index_hash.push_back(pair<int, int>(index_hash.size() + 1, 1));
						dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = index_hash.size();
					}
					else{
						//cout << accumulate(index_sort.begin(), index_sort.end(), 0) << endl;
						min_id = min_id_default;
						for (int i = 0; i < 4; ++i){
							if (index_sort[i] == 0)continue;
							if (min_id == min_id_default){
								min_id = index_sort[i];
							}
							else{
								index_hash[index_sort[i] - 1].first = min_id;
								index_hash[index_sort[i] - 1].second++;
								dst_tmp.data[y * dst_tmp.step + x * dst_tmp.elemSize()] = min_id;
							}
						}
					}
				}
			}
				//cout << min_id << endl;			}
		}
	}

	//if (src_channel == 1)  bgr_img[0].copyTo(dst);	// 単色
	//else cv::merge(bgr_img, 3, dst);				// 三色(BGR)
	std::sort(index_hash.begin(), index_hash.end());
	//index_hash.erase(std::unique(index_hash.begin(), index_hash.end()), index_hash.end());
	vector<pair<int, int>> index_out;
	vector<pair<int, int>> index_table;
	int now_id = 0;
	int step = 1;
	for (auto v : index_hash){
		if (now_id != v.first){
			now_id = v.first;
			index_out.push_back(pair<int, int>(step, v.second));
			index_table.push_back(pair<int, int>(v.first, step));
			step++;
		}
		else{
			index_out[step-2].second += v.second;
		}
	}

	for (auto v : index_out){
		cout << v.first << " , " << v.second << endl;
	}
	cout << endl <<  index_out.size() << endl;

	Mat dst_test;
	threshold(dst_tmp, dst_test, 0, 255, cv::THRESH_BINARY);
	imshow("aaa", dst_test);
	waitKey(0);


	Mat dst_thresh, dst_thresh2, dst_thresh3;
	Mat dst_out = Mat::zeros(dst_tmp.size(),CV_8UC1);
	const int area_thresh = 100;
	for (int i = 0; i < index_out.size(); ++i){
		if (index_out[i].second > area_thresh){
			threshold(dst_tmp, dst_thresh, index_table[i].first-1, 255, cv::THRESH_TOZERO);
			threshold(dst_tmp, dst_thresh2, index_table[i].first, 255, cv::THRESH_TOZERO);
			absdiff(dst_thresh2, dst_thresh, dst_thresh3);
			threshold(dst_thresh3, dst_thresh3, 1, index_table[i].second, cv::THRESH_BINARY);
			//imshow("test", dst_thresh3);
			//waitKey(0);
			add(dst_out, dst_thresh3, dst_out);
		}
	}

	threshold(dst_out, dst_tmp, 0, 255, cv::THRESH_BINARY);

	imshow("aaa", dst_tmp);
	waitKey(0);
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

	std::string name;
	std::string path;
	std::string input;
	if (argc > 1){
		input = argv[1];
		std::string::size_type pos = input.find_last_of('L');
		path = input.substr(0, pos);
	}
	else{
		path = "../picture/Japan";
	}

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	//Mat img1 = imread("../picture/77L.jpg", 1);path+number+ "LEFT"+frameNum+".bmp"
	Mat img1 = imread(path+"L.jpg", 1);
	Mat img2 = imread(path+"R.jpg", 1);
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

		//angle = 0.0, scale = 1.0;

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
		Mat img1_tmp;
		//GaussianBlur(img1, img1_tmp, Size(5, 5), 1.5);
		//GaussianBlur(img2_out_resize, img2_out_resize, Size(5, 5), 1.5);
		absdiff(img1, img2_out_resize, diff);
		cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), 1);
		cv::dilate(diff, diff, cv::Mat(), cv::Point(-1, -1), 4);
		cv::erode(diff, diff, cv::Mat(), cv::Point(-1, -1), 3);
		imshow("diff", diff);

		Mat diff_gray;
		cvtColor(diff, diff_gray, cv::COLOR_RGB2GRAY);
		//adaptiveThreshold(diff_gray, diff_gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 1);

		//GaussianBlur(diff_gray, diff_gray, Size(7, 7), 1.5);
		//Canny(diff_gray, diff_gray, 10, 50);
		//threshold(diff_gray, diff_gray, 10, 255, cv::THRESH_BINARY);
		imshow("diff_gray", diff_gray);


		//bitwise_and(diff_gray, old_gray, and_gray);
		//old_gray = diff_gray.clone();
		//imshow("and_gray", and_gray);


		//LabelingArea(diff_gray, diff, 10);
		
		const int ch_width = 260;
		const int sch = diff_gray.channels();
		Mat hist_img(Size(ch_width * sch, 200), CV_8UC3, Scalar::all(255));

		vector<MatND> hist(3);
		const int hist_size = 256;
		const int hdims[] = { hist_size };
		const float hranges[] = { 0, 256 };
		const float* ranges[] = { hranges };
		double max_val = .0;

		if (sch == 1) {
			// (3a)if the source image has single-channel, calculate its histogram
			calcHist(&diff_gray, 1, 0, Mat(), hist[0], 1, hdims, ranges, true, false);
			minMaxLoc(hist[0], 0, &max_val);
		}
		else {
			// (3b)if the souce image has multi-channel, calculate histogram of each plane
			for (int i = 0; i<sch; ++i) {
				calcHist(&diff_gray, 1, &i, Mat(), hist[i], 1, hdims, ranges, true, false);
				double tmp_val;
				minMaxLoc(hist[i], 0, &tmp_val);
				max_val = max_val < tmp_val ? tmp_val : max_val;
			}
		}

		// (4)scale and draw the histogram(s)
		Scalar color = Scalar::all(100);
		for (int i = 0; i<sch; i++) {
			if (sch == 3)
				color = Scalar((0xaa << i * 8) & 0x0000ff, (0xaa << i * 8) & 0x00ff00, (0xaa << i * 8) & 0xff0000, 0);
			hist[i].convertTo(hist[i], hist[i].type(), max_val ? 200. / max_val : 0., 0);
			for (int j = 0; j<hist_size; ++j) {
				int bin_w = saturate_cast<int>((double)ch_width / hist_size);
				rectangle(hist_img,
					Point(j*bin_w + (i*ch_width), hist_img.rows),
					Point((j + 1)*bin_w + (i*ch_width), hist_img.rows - saturate_cast<int>(hist[i].at<float>(j))),
					color, -1);
			}
		}
		imshow("Histogram", hist_img);

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

