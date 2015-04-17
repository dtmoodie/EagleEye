#include "nodes/ImgProc/FFT.h"
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")

void FFT::Init(bool firstInit)
{
    updateParameter("DFT rows flag", false);
    updateParameter("DFT scale flag", false);
    updateParameter("DFT inverse flag", false);
    updateParameter("DFT real output flag", false);
    updateParameter("Desired output", int(-1));
    updateParameter("Log scale", true);
}

cv::cuda::GpuMat FFT::doProcess(cv::cuda::GpuMat &img)
{
    if(img.empty())
        return img;
    int rows = cv::getOptimalDFTSize(img.rows);
    int cols = cv::getOptimalDFTSize(img.cols);
    cv::cuda::GpuMat padded;
    cv::cuda::copyMakeBorder(img,padded, 0, rows - img.rows, 0, cols - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
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
        img.convertTo(floatImg,CV_MAKETYPE(CV_32F,img.channels()));
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
    cv::cuda::dft(floatImg,dest,img.size(),flags);
    int channel = getParameter<int>(4)->data;
    if(parameters[4]->changed)
    {
        log(Status, channel == 0 ? "Magnitude" : "Phase");
        parameters[4]->changed = false;
    }
    if(channel != -1)
    {
        if(channel == 0)
        {

            cv::cuda::magnitude(dest,dest);
            if(getParameter<bool>(5)->data)
            {
                // Convert to log scale
                cv::cuda::add(dest,cv::Scalar::all(1), dest);
                cv::cuda::log(dest,dest);
            }
        }
        if(channel == 1)
        {
            std::vector<cv::cuda::GpuMat> channels;
            cv::cuda::split(dest,channels);

            cv::cuda::phase(channels[0],channels[1],dest);
        }
    }
    return dest;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(FFT);
