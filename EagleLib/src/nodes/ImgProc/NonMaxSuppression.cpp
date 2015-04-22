#include "nodes/ImgProc/NonMaxSuppression.h"
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;

void MinMax::Init(bool firstInit)
{
    updateParameter<double>("Min value", 0.0, Parameter::Output);
    updateParameter<double>("Max value", 0.0, Parameter::Output);
}

cv::cuda::GpuMat MinMax::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    cv::cuda::minMax(img, &getParameter<double>(0)->data, &getParameter<double>(1)->data);

    return img;
}
void Threshold::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<double>("Input max"); // 0
        addInputParameter<double>("Input min"); // 1
        updateParameter<double>("Replace Value", 255.0); // 2
        updateParameter<double>("Max", 0.0); // 3
        updateParameter<double>("Min", 0.0); // 4
        updateParameter<bool>("Two sided", false, Parameter::Control, "If true, min and max are used to define a threshold range");
        updateParameter<bool>("Truncate", false, Parameter::Control, "If true, threshold to original value, not replace value");
        updateParameter<bool>("Inverse", false, Parameter::Control, "If true, inverse threshold is applied, ie values greater than max and less than min pass");
        updateParameter<cv::cuda::GpuMat>("Mask", cv::cuda::GpuMat(), Parameter::Output);
    }
}

cv::cuda::GpuMat Threshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    cv::cuda::GpuMat mask;
    bool inverse = getParameter<bool>(7)->data;
    bool truncate = getParameter<bool>(6)->data;
    if(getParameter<bool>(6)->data)
    {
        // Two sided means max value will also be used to find an upper bound


    }

	return img;
}

void NonMaxSuppression::Init(bool firstInit)
{
    updateParameter<int>("Size", 3);
    addInputParameter<cv::cuda::GpuMat>("Mask");
}

cv::cuda::GpuMat NonMaxSuppression::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    // initialise the block mask and destination
    const int M = img.rows;
    const int N = img.cols;
    cv::Mat src(img);
    int sz = getParameter<int>("Size")->data;

    cv::Mat mask;
    if(auto maskParam = getParameterPtr<cv::cuda::GpuMat>(parameters[1]))
        if(!maskParam->empty())
            maskParam->download(mask);
    const bool masked = !mask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*sz+1,2*sz+1));
    cv::Mat dst = cv::Mat_<uint8_t>::zeros(src.size());

    // iterate over image blocks
    for (int m = 0; m < M; m+=sz+1) {
        for (int n = 0; n < N; n+=sz+1) {
            cv::Point  ijmax;
            double vcmax, vnmax;

            // get the maximal candidate within the block
            cv::Range ic(m, cv::min(m+sz+1,M));
            cv::Range jc(n, cv::min(n+sz+1,N));

            cv::minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : cv::noArray());


            cv::Point cc = ijmax + cv::Point(jc.start,ic.start);

            // search the neighbours centered around the candidate for the true maxima
            cv::Range in(cv::max(cc.y-sz,0), cv::min(cc.y+sz+1,M));
            cv::Range jn(cv::max(cc.x-sz,0), cv::min(cc.x+sz+1,N));

            // mask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()), cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start, cv::min(ic.start-in.start+sz+1, in.size()));
            cv::Range jis(jc.start-jn.start, cv::min(jc.start-jn.start+sz+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));

            cv::minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
            cv::Point cn = ijmax + cv::Point(jn.start, in.start);

            // if the block centre is also the neighbour centre, then it's a local maxima
            if (vcmax > vnmax) {
                dst.at<uint8_t>(cc.y, cc.x) = 255;
            }
        }
    }
    return cv::cuda::GpuMat(dst);
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(NonMaxSuppression)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MinMax)
