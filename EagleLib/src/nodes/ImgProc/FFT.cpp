#include "nodes/ImgProc/FFT.h"
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")

void FFT::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("DFT rows flag", false);        // 0
        updateParameter("DFT scale flag", false);       // 1
        updateParameter("DFT inverse flag", false);     // 2
        updateParameter("DFT real output flag", false); // 3
        updateParameter("Desired output", int(-1));     // 4
        updateParameter("Log scale", true);             // 5
        updateParameter<cv::cuda::GpuMat>("Magnitude", cv::cuda::GpuMat(), Parameter::Output);  // 6
        updateParameter<cv::cuda::GpuMat>("Phase", cv::cuda::GpuMat(), Parameter::Output);      // 7
    }
    updateParameter("Use optimized size",false);
}

cv::cuda::GpuMat FFT::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(img.empty())
        return img;
    int rows = cv::getOptimalDFTSize(img.rows);
    int cols = cv::getOptimalDFTSize(img.cols);
    cv::cuda::GpuMat padded;
    if(getParameter<bool>("Use optimized size")->data)
        cv::cuda::copyMakeBorder(img,padded, 0, rows - img.rows, 0, cols - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0), stream);
    else
        padded = img;
    img = padded;
    if(img.channels() > 2)
    {
        std::stringstream ss;
        ss << "Too many channels, can only handle 1 or 2 channel input. Input has ";
        ss << img.channels() << " channels.";
        log(Warning, ss.str());
        return img;
    }
    cv::cuda::GpuMat floatImg;
    if(img.depth() != CV_32F)
        img.convertTo(floatImg,CV_MAKETYPE(CV_32F,img.channels()), stream);
    else
        floatImg = img;
    cv::cuda::GpuMat dest;
    int flags = 0;
    if(getParameter<bool>(0)->data)
        flags = flags | cv::DFT_ROWS;
    if(getParameter<bool>(1)->data)
        flags = flags | cv::DFT_SCALE;
    if(getParameter<bool>(2)->data)
        flags = flags | cv::DFT_INVERSE;
    if(getParameter<bool>(3)->data)
        flags = flags | cv::DFT_REAL_OUTPUT;
    cv::cuda::dft(floatImg,dest,img.size(),flags, stream);
    int channel = getParameter<int>(4)->data;
    if(parameters[4]->changed)
    {
        log(Status, channel == 0 ? "Magnitude" : "Phase");
        parameters[4]->changed = false;
    }
    cv::cuda::GpuMat magnitude, phase;
    if(channel == 0 || parameters[6]->subscribers != 0)
    {
        cv::cuda::magnitude(dest,magnitude, stream);
        if(getParameter<bool>(5)->data)
        {
            // Convert to log scale
            cv::cuda::add(magnitude,cv::Scalar::all(1), magnitude, cv::noArray(), -1, stream);
            cv::cuda::log(magnitude,magnitude, stream);
        }

        updateParameter(6,magnitude);
    }
    if(channel == 1 || parameters[7]->subscribers != 0)
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(dest,channels, stream);
        cv::cuda::phase(channels[0],channels[1],phase, false, stream);
        updateParameter(7, phase);
    }
    if(channel == 1)
        dest = phase;
    if(channel == 0)
        dest = magnitude;
    return dest;
}
cv::Mat getShiftMat(cv::Size matSize)
{
    cv::Mat shift(matSize, CV_32F);
    for(int y = 0; y < matSize.height; ++y)
    {
        for(int x = 0; x < matSize.width; ++x)
        {
            shift.at<float>(y,x) = 1.0 - 2.0 * ((x+y)&1);
        }
    }

    return shift;
}

void FFTPreShiftImage::Init(bool firstInit)
{

}

cv::cuda::GpuMat FFTPreShiftImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(d_shiftMat.size() != img.size())
    {
        d_shiftMat.upload(getShiftMat(img.size()), stream);
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat,img, result,1,-1,stream);
    return result;
}
void FFTPostShift::Init(bool firstInit)
{

}

cv::cuda::GpuMat FFTPostShift::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(d_shiftMat.size() != img.size())
    {
        d_shiftMat.upload(getShiftMat(img.size()), stream);
        std::vector<cv::cuda::GpuMat> channels;
        channels.push_back(d_shiftMat);
        channels.push_back(d_shiftMat);
        cv::cuda::merge(channels, d_shiftMat, stream);
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat,img, result, 1 / float(img.size().area()), -1, stream);
    return result;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(FFT);
NODE_DEFAULT_CONSTRUCTOR_IMPL(FFTPreShiftImage);
NODE_DEFAULT_CONSTRUCTOR_IMPL(FFTPostShift);
