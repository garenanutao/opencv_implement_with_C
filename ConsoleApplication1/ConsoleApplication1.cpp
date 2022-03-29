#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

void ResizeBilinear(Mat& mat_src, Mat& mat_dst, double scale_x, double scale_y)
{
    uchar* data_dst = mat_dst.data;
    size_t step_dst = mat_dst.step;
    uchar* data_src = mat_src.data;
    size_t step_src = mat_src.step;
    int width_dst = mat_dst.cols;
    int height_dst = mat_dst.rows;
    int elementSize_dst = mat_dst.elemSize();
    int width_src = mat_src.cols;
    int height_src = mat_src.rows;
    int elementSize_src = mat_src.elemSize();

    for (int j = 0; j < height_dst; ++j)
    {
        float fy = (float)((j + 0.5f) * scale_y - 0.5f);
        int iy = floor(fy);
        fy -= iy;

        iy = iy >= (height_src - 2) ? (height_src - 2) : iy;
        iy = iy < 0 ? 0 : iy;
        fy = iy >= (height_src - 2) ? 0 : fy;
        fy = iy < 0 ? 0 : fy;

        short ftoi_fy0 = (1.f-fy) * 2048;
        short ftoi_fy1 = 2048 - ftoi_fy0;

        for (int i = 0; i < width_dst; ++i) 
        {
            float fx = (float)((i + 0.5f) * scale_x - 0.5f);
            int ix = floor(fx);
            fx -= ix;

            ix = ix >= (width_src - 2) ? (width_src - 2) : ix;
            ix = ix < 0 ? 0 : ix;
            fx = ix >= (width_src - 2) ? 0 : fx;
            fx = ix < 0 ? 0 : fx;

            short ftoi_fx0 = (1.f - fx) * 2048;
            short ftoi_fx1 = 2048 - ftoi_fx0;


            for (int k = 0; k < mat_src.channels(); ++k)
            {
                *(data_dst + j * step_dst + i * elementSize_dst + k) = (*(data_src + iy * step_src + ix * elementSize_src + k) * ftoi_fx0 * ftoi_fy0
                    + *(data_src + (iy+1) * step_src + ix * elementSize_src + k) * ftoi_fx0 * ftoi_fy1
                    + *(data_src + iy * step_src + (ix+1) * elementSize_src + k) * ftoi_fx1 * ftoi_fy0
                    + *(data_src + (iy+1) * step_src + (ix+1) * elementSize_src + k) * ftoi_fx1 * ftoi_fy1) >> 22;
            }
        }
    }

    cv::imwrite("bilinear.jpg", mat_dst);
}

void ResizeBicubic(Mat& mat_src, Mat& mat_dst, double scale_x, double scale_y)
{
    uchar* data_dst = mat_dst.data;
    size_t step_dst = mat_dst.step;
    uchar* data_src = mat_src.data;
    size_t step_src = mat_src.step;
    int width_dst = mat_dst.cols;
    int height_dst = mat_dst.rows;
    int elementSize_dst = mat_dst.elemSize();
    int width_src = mat_src.cols;
    int height_src = mat_src.rows;
    int elementSize_src = mat_src.elemSize();
    

    for (int j = 0; j < height_dst; ++j)
    {
        float fy = (float)((j+0.5f) * scale_y -0.5f);
        int iy = floor(fy);
        fy -= iy;
        
        iy = iy >= (height_src - 3) ? (height_src -3) : iy;
        iy = iy < 1 ? 1 : iy;
        fy = iy >= (height_src - 3) ? 0 : fy;
        fy = iy < 1 ? 0 : fy;
        
        const float A = -0.5f;

        float ftoi_fy0 = ((A * (1.f + fy) - 5 * A) * (1.f + fy) + 8 * A) * (1.f + fy) - 4 * A;
        float ftoi_fy1 = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        float ftoi_fy2 = ((A + 2) * (1.f-fy) - (A + 3)) * (1.f - fy) * (1.f - fy) + 1;
        float ftoi_fy3 = 1.f - ftoi_fy0 - ftoi_fy1 - ftoi_fy2;
       
        short ftos_fy0 = cv::saturate_cast<short>(ftoi_fy0 * 2048);
        short ftos_fy1 = cv::saturate_cast<short>(ftoi_fy1 * 2048);
        short ftos_fy2 = cv::saturate_cast<short>(ftoi_fy2 * 2048);
        short ftos_fy3 = cv::saturate_cast<short>(ftoi_fy3 * 2048);


        for (int i = 0; i < width_dst; ++i)
        {
            float fx = (float)((i+0.5f) * scale_x-0.5f);
            int ix = floor(fx);
            fx -= ix;
            
            ix = ix >= (width_src - 3) ? (width_src - 3) : ix;
            ix = ix < 1 ? 1 : ix;
            fx = ix >= (width_src - 3) ? 0 : fx;
            fx = ix < 1 ? 0 : fx;
            
            float ftoi_fx0 = ((A * (1.f + fx) - 5 * A) * (1.f + fx) + 8 * A) * (1.f + fx) - 4 * A;
            float ftoi_fx1 = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
            float ftoi_fx2 = ((A + 2) * (1.f - fx) - (A + 3)) * (1.f - fx) * (1.f - fx) + 1;
            float ftoi_fx3 = 1.f - ftoi_fx0 - ftoi_fx1 - ftoi_fx2;

            short ftos_fx0 = cv::saturate_cast<short>(ftoi_fx0 * 2048);
            short ftos_fx1 = cv::saturate_cast<short>(ftoi_fx1 * 2048);
            short ftos_fx2 = cv::saturate_cast<short>(ftoi_fx2 * 2048);
            short ftos_fx3 = cv::saturate_cast<short>(ftoi_fx3 * 2048);
            

            for (int k = 0; k < mat_src.channels(); ++k)
            {
                *(data_dst + j * step_dst + i * elementSize_dst + k) = abs((*(data_src + (iy-1) * step_src + (ix-1) * elementSize_src + k) * ftos_fy0 * ftos_fx0
                    + *(data_src + (iy - 1) * step_src + ix * elementSize_src + k) * ftos_fy0 * ftos_fx1
                    + *(data_src + (iy - 1) * step_src + (ix+1) * elementSize_src + k) * ftos_fy0 * ftos_fx2
                    + *(data_src + (iy - 1) * step_src + (ix+2) * elementSize_src + k) * ftos_fy0 * ftos_fx3

                    + *(data_src + iy * step_src + (ix-1) * elementSize_src + k) * ftos_fy1 * ftos_fx0
                    + *(data_src + iy * step_src + ix * elementSize_src + k) * ftos_fy1 * ftos_fx1
                    + *(data_src + iy * step_src + (ix+1) * elementSize_src + k) * ftos_fy1 * ftos_fx2
                    + *(data_src + iy * step_src + (ix+2) * elementSize_src + k) * ftos_fy1 * ftos_fx3

                    + *(data_src + (iy+1) * step_src + (ix - 1) * elementSize_src + k) * ftos_fy2 * ftos_fx0
                    + *(data_src + (iy+1) * step_src + ix * elementSize_src + k) * ftos_fy2 * ftos_fx1
                    + *(data_src + (iy+1) * step_src + (ix + 1) * elementSize_src + k) * ftos_fy2 * ftos_fx2
                    + *(data_src + (iy+1) * step_src + (ix + 2) * elementSize_src + k) * ftos_fy2 * ftos_fx3

                    + *(data_src + (iy+2) * step_src + (ix - 1) * elementSize_src + k) * ftos_fy3 * ftos_fx0
                    + *(data_src + (iy+2) * step_src + ix * elementSize_src + k) * ftos_fy3 * ftos_fx1
                    + *(data_src + (iy+2) * step_src + (ix + 1) * elementSize_src + k) * ftos_fy3 * ftos_fx2
                    + *(data_src + (iy+2) * step_src + (ix + 2) * elementSize_src + k) * ftos_fy3 * ftos_fx3) >> 22);
            }
        }
    }

    cv::imwrite("bicubic.jpg", mat_dst);
}


int main(int argc, char** argv) {

    String image_path = "test.png";//"lena_std.tif";
	Mat image = imread(image_path);

	if (!image.data) {
        printf("cannot open image!\n");
		return -1;
	}

    float scale_x = 1.8f;
    float scale_y = 1.8f;
    cv::Size scale_size(image.size().width*scale_x, image.size().height*scale_y);

    Mat resize_image = cv::Mat{scale_size, image.type(), cv::Scalar{0,0,0} };
    ResizeBilinear(image, resize_image, 1.f/scale_x, 1.f / scale_y);

    Mat matDst2;
    cv::resize(image, matDst2, scale_size, 0, 0, cv::INTER_LINEAR);
    cv::imwrite("bilinear_sample.jpg", matDst2);

    resize_image = cv::Mat{ scale_size, image.type(), cv::Scalar{0,0,0} };
    ResizeBicubic(image, resize_image, 1.f / scale_x, 1.f / scale_y);

    Mat matDst3;
    cv::resize(image, matDst3, scale_size, 0, 0, cv::INTER_CUBIC);
    cv::imwrite("bicubic_sample.jpg", matDst3);


	//namedWindow("Display Window", 1);
	//imshow("Display Window", image);
	//waitKey(0);
	return 0;
}