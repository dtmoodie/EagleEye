#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "WhiteBalance.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <iostream>
using namespace aq::nodes;
using namespace cv;

bool WhiteBalance::processImpl()
{
    cv::cuda::GpuMat output;
    auto lower = cv::Scalar(lower_blue, lower_green, lower_red);
    auto upper = cv::Scalar(upper_blue, upper_green, upper_red);
    applyWhiteBalance(input->getGpuMat(stream()), output, lower, upper, rois, weight, dtype, stream());
    output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
    /*const cv::Mat& in = input->getMat(stream());
    cv::Mat out;
    stream().waitForCompletion();
    CV_Assert(in.channels() == 3);
    int low[3];
    int high[3];
    std::vector<Mat> tmpsplit; split(in,tmpsplit);
    if(rois.empty())
     rois =
     {
        cv::Rect2f(0.2, 0.2, 0.1, 0.1),
        cv::Rect2f(0.7, 0.7, 0.1, 0.1),
        cv::Rect2f(0.1, 0.7, 0.1, 0.1),
        cv::Rect2f(0.7, 0.1, 0.1, 0.1),
        cv::Rect2f(0.45, 0.45, 0.1, 0.1)
     };
    if(weight.empty())
        weight = { 1, 1, 1, 1, 1};
    float sum = 0;
    for(int i = 0; i < weight.size(); ++i)
    {
        sum += weight[i];
    }
    for(int i = 0; i < weight.size(); ++i)
    {
        weight[i] /= sum;
    }
    int rows = in.rows;
    int cols = in.cols;
    for(int i=0;i<3;i++)
    {
        //find the low and high precentile values (based on the input percentile)
        float lowval = 0, highval = 0;
        for(int j = 0; j < rois.size(); ++j)
        {
            Mat flat;
            const cv::Rect2f& roi = rois[j];
            tmpsplit[i](cv::Rect(roi.x*cols, roi.y*rows, roi.width*cols, roi.height*rows)).copyTo(flat);
            flat = flat.reshape(1,1);

            cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

            if(flat.depth() == CV_8U)
            {
                lowval += weight[j] * flat.at<uchar>(cvFloor(((float)flat.cols) * lower_percent));
                highval += weight[j] * flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - upper_percent)));
            }else
            {
                if(flat.depth() == CV_16U)
                {
                    lowval += weight[j] * flat.at<ushort>(cvFloor(((float)flat.cols) * lower_percent));
                    highval += weight[j] * flat.at<ushort>(cvCeil(((float)flat.cols) * (1.0 - upper_percent)));
                }
            }
        }
        low[i] = lowval;
        high[i] = highval;
    }

    for(int i = 0; i < 3; ++i)
    {
        tmpsplit[i].setTo(low[i],tmpsplit[i] < low[i]);
        tmpsplit[i].setTo(high[i],tmpsplit[i] > high[i]);

        //scale the channel
        normalize(tmpsplit[i],tmpsplit[i],min,max,NORM_MINMAX, dtype);
    }

    merge(tmpsplit,out);
    output_param.updateData(out, input_param.getTimestamp(), _ctx.get());*/
    return true;
}

MO_REGISTER_CLASS(WhiteBalance)

bool StaticWhiteBalance::processImpl()
{
    const cv::cuda::GpuMat& in = input->getGpuMat(stream());
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(in, channels, stream());
    for (int i = 0; i < 3; ++i)
    {
        cv::cuda::threshold(channels[i], channels[i], high[i], high[i], cv::THRESH_TRUNC, stream());
        cv::cuda::GpuMat mask;
        cv::cuda::threshold(channels[i], mask, low[i], 255, cv::THRESH_BINARY_INV, stream());
        channels[i].setTo(low[i], mask, stream());
        cv::cuda::normalize(channels[i], channels[i], min, max, cv::NORM_MINMAX, dtype, cv::noArray(), stream());
    }
    cv::cuda::GpuMat output;
    cv::cuda::merge(channels, output, stream());
    output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(StaticWhiteBalance)

bool WhiteBalanceMean::processImpl()
{
    const cv::cuda::GpuMat& in = input->getGpuMat(stream());
    std::vector<cv::cuda::GpuMat> channels;
    channels.resize(in.channels());
    for (int i = 0; i < in.channels(); ++i)
    {
        cv::cuda::createContinuous(in.size(), in.depth(), channels[i]);
    }
    cv::cuda::split(in, channels, stream());
    cv::cuda::GpuMat mean;
    cv::cuda::createContinuous(channels.size(), 1, CV_64F, mean);
    for (int i = 0; i < channels.size(); ++i)
    {
        // cv::cuda::meanStdDev(channels[i], mean.row(i), stream());
        channels[i] = channels[i].reshape(1, 1);
        cv::cuda::reduce(channels[i], mean.row(i), 1, CV_REDUCE_AVG, CV_64F, stream());
    }

    cv::Mat h_mean(mean);
    double max = h_mean.at<double>(0, 0);
    for (int i = 1; i < h_mean.rows; ++i)
    {
        max = std::max(h_mean.at<double>(i, 0), max);
    }
    cv::Scalar gain(max / h_mean.at<double>(0, 0), max / h_mean.at<double>(1, 0), max / h_mean.at<double>(2, 0));
    h_m.create(3, 3, CV_32F);

    // Saturation adjustment from http://www.siliconimaging.com/RGB%20Bayer.htm
    std::cout << h_mean.at<double>(0, 0) << " " << h_mean.at<double>(1, 0) << " " << h_mean.at<double>(2, 0)
              << std::endl;
    float m00 = (0.299 + 0.701 * K);
    float m10 = (0.299 * (1 - K));
    float m20 = (0.299 * (1 - K));

    float m01 = (0.587 * (1 - K));
    float m11 = (0.587 + 0.413 * K);
    float m21 = (0.587 * (1 - K));

    float m02 = (0.114 * (1 - K));
    float m12 = (0.114 * (1 - K));
    float m22 = (0.114 + 0.886 * K);

    float b = gain.val[0];
    float g = gain.val[1];
    float r = gain.val[2];

    // Weird matrix order due to opencv BGR ordering
    h_m.at<float>(0, 0) = b * m22;
    h_m.at<float>(1, 0) = b * m12;
    h_m.at<float>(2, 0) = b * m02;

    h_m.at<float>(0, 1) = g * m21;
    h_m.at<float>(1, 1) = g * m11;
    h_m.at<float>(2, 1) = g * m01;

    h_m.at<float>(0, 2) = r * m20;
    h_m.at<float>(1, 2) = r * m10;
    h_m.at<float>(2, 2) = r * m00;

    d_m.upload(h_m, stream());

    cv::cuda::GpuMat output;
    in.copyTo(output, stream());
    colorCorrect(output, d_m, stream());
    // cv::cuda::multiply(in, gain, output, 1, -1, stream());
    output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(WhiteBalanceMean);
#endif