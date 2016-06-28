#include "nodes/Utility/Frame.h"
#include "EagleLib/rcc/external_includes/cv_cudawarping.hpp"
#include "EagleLib/rcc/external_includes/cv_cudaarithm.hpp"
#include <EagleLib/Qualifiers.hpp>
#include <parameters/ParameteredObjectImpl.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;
void FrameRate::NodeInit(bool firstInit)
{

}

cv::cuda::GpuMat FrameRate::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    prevTime = currentTime;
    updateParameter<double>("Framerate", 1000.0/delta.total_milliseconds())->type =  Parameters::Parameter::State;
    return img;
}

void FrameLimiter::NodeInit(bool firstInit)
{
    updateParameter<double>("Framerate", 60.0);
}

cv::cuda::GpuMat FrameLimiter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta(currentTime - lastTime);
    lastTime = currentTime;
    int goalTime = 1000.0 / *getParameter<double>(0)->Data();
    if (delta.total_milliseconds() < goalTime)
    {
        //boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        boost::this_thread::sleep_for(boost::chrono::milliseconds(goalTime - delta.total_milliseconds()));
    }
    return img;
}
void CreateMat::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        Parameters::EnumParameter dataType;
        dataType.addEnum(ENUM(CV_8U));
        dataType.addEnum(ENUM(CV_8S));
        dataType.addEnum(ENUM(CV_16U));
        dataType.addEnum(ENUM(CV_16S));
        dataType.addEnum(ENUM(CV_32S));
        dataType.addEnum(ENUM(CV_32F));
        dataType.addEnum(ENUM(CV_64F));
        updateParameter("Datatype", dataType);
        updateParameter("Channels", int(1));
        updateParameter("Width", 1920);
        updateParameter("Height", 1080);
        updateParameter("Fill", cv::Scalar::all(0));
    }
    _parameters[0]->changed = true;
}

cv::cuda::GpuMat CreateMat::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(_parameters[0]->changed ||
       _parameters[1]->changed ||
       _parameters[2]->changed ||
       _parameters[3]->changed ||
       _parameters[4]->changed )
    {
        auto typeParam = getParameter<Parameters::EnumParameter>(0);
        int selection = typeParam->Data()->currentSelection;
        int dtype = typeParam->Data()->values[selection];
        createdMat = cv::cuda::GpuMat(*getParameter<int>(3)->Data(),
                    *getParameter<int>(2)->Data(),
                    dtype, *getParameter<cv::Scalar>(4)->Data());
        updateParameter("Output", createdMat)->type = Parameters::Parameter::Output;
        _parameters[0]->changed = false;
        _parameters[1]->changed = false;
        _parameters[2]->changed = false;
        _parameters[3]->changed = false;
        _parameters[4]->changed = false;
    }
    return img;
}
void SetMatrixValues::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input image");
        addInputParameter<cv::cuda::GpuMat>("Input mask");
        updateParameter("Replace value", cv::Scalar(0,0,0));
    }
    qualifiersSetup = false;
}

cv::cuda::GpuMat SetMatrixValues::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    TIME
        cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat>(0)->Data();
    if(input == nullptr)
        input = &img;
    TIME
    if(_parameters[0]->changed || qualifiersSetup == false)
    {
        boost::function<bool(Parameters::Parameter*)> f = GpuMatQualifier::get(input->cols, input->rows, 1, CV_8U);
        updateInputQualifier<cv::cuda::GpuMat>(1, f);
    }
    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>(1)->Data();

    if(mask && mask->size() == input->size())
    {
        TIME
            input->setTo(*getParameter<cv::Scalar>(2)->Data(), *mask, stream);
        TIME
    }else
    {
        TIME
        input->setTo(*getParameter<cv::Scalar>(2)->Data(), stream);
        TIME
    }
    return *input;
}
void Resize::NodeInit(bool firstInit)
{
    updateParameter("Width", int(224));
    updateParameter("Height", int(224));
}
cv::cuda::GpuMat Resize::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto buf = bufferPool.getFront();
    cv::cuda::resize(img, buf->data, cv::Size(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data()), 0.0, 0.0, 1, stream);
    return buf->data;
}
void Subtract::NodeInit(bool firstInit)
{
    updateParameter("Value to subtract", cv::Scalar(0, 0, 0));
}
cv::cuda::GpuMat Subtract::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::subtract(img, *getParameter<cv::Scalar>(0)->Data(), img, cv::noArray(), -1, stream);
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(SetMatrixValues, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FrameRate, Utility)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FrameLimiter, Utility)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CreateMat, Image, Source)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Resize, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Subtract, Image, Processing)
