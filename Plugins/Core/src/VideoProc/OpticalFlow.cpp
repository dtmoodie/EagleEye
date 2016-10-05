#include "OpticalFlow.h"
#include "EagleLib/nodes/VideoProc/Tracking.hpp"
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudawarping.hpp>


using namespace EagleLib;
using namespace EagleLib::Nodes;

#include "ObjectInterfacePerModule.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cudaoptflow")
#endif


NODE_DEFAULT_CONSTRUCTOR_IMPL(SparsePyrLKOpticalFlow, Image, Extractor)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BroxOpticalFlow, Image, Extractor)
NODE_DEFAULT_CONSTRUCTOR_IMPL(DensePyrLKOpticalFlow, Image, Extractor)


void DensePyrLKOpticalFlow::NodeInit(bool firstInit)
{
    opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13, 13), 3, 30, false); 
    updateParameter<int>("Window Size", 13);
    updateParameter<int>("Pyramid levels", 3);
    updateParameter<int>("Iterations", 30);
    updateParameter<bool>("Use initial flow", false);
}
cv::cuda::GpuMat DensePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if (_parameters[0]->changed || _parameters[1]->changed || _parameters[2]->changed || _parameters[3]->changed)
    {
        int size = *getParameter<int>("Window Size")->Data();
        int levels = *getParameter<int>("Pyramid levels")->Data();
        int iters = *getParameter<int>("Iterations")->Data();
        opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(size, size), levels, iters, false);
        _parameters[0]->changed = false;
        _parameters[1]->changed = false; 
        _parameters[2]->changed = false;
        _parameters[3]->changed = false;
    }
    

    if (img.channels() != 1)
    {
        cv::cuda::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY, 0, stream);
    }
    else
    {
        greyImg = img;
    }
    if (prevGreyImg.empty())
    {
        greyImg.copyTo(prevGreyImg, stream);
        return img;
    }
    opt_flow->calc(prevGreyImg, greyImg, flow, stream);
    //cv::Mat field(flow);
    //cv::Mat prev(prevGreyImg);
    //cv::Mat curr(greyImg);
    greyImg.copyTo(prevGreyImg, stream);
    updateParameter("FlowField", flow);
    return img;
}
void SparsePyrLKOpticalFlow::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input Points");
        updateParameter("Window size", int(21));
        updateParameter("Max Levels", int(3));
        updateParameter("Iterations", int(30));
        updateParameter("Use initial flow", true);
        updateParameter("Enabled", false);
        _parameters[1]->changed = true;
    }
    updateParameter<boost::function<void(cv::cuda::GpuMat&, cv::cuda::GpuMat&, size_t, cv::cuda::Stream&)>>("Set reference callback", boost::bind(&SparsePyrLKOpticalFlow::set_reference, this, _1,_2,_3,_4));
}
void SparsePyrLKOpticalFlow::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(optFlow);
}
void SparsePyrLKOpticalFlow::set_reference(cv::cuda::GpuMat& ref_image, cv::cuda::GpuMat& ref_points, size_t frame_number, cv::cuda::Stream& stream)
{

}

void build_pyramid(std::vector<cv::cuda::GpuMat>& pyramid, int levels, cv::cuda::Stream& stream)
{
    CV_Assert(pyramid.size());
    CV_Assert(!pyramid[0].empty());
    pyramid.resize(levels);
    for (int level = 1; level < levels; ++level)
    {
        cv::cuda::pyrDown(pyramid[level - 1], pyramid[level], stream);
    }
}

cv::cuda::GpuMat SparsePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(_parameters[1]->changed ||
       _parameters[2]->changed ||
       _parameters[3]->changed ||
       _parameters[4]->changed)
    {
        int winSize = *getParameter<int>(1)->Data();
        int levels = *getParameter<int>(2)->Data();
        int iters = *getParameter<int>(3)->Data();
        bool useInital = *getParameter<bool>(4)->Data();
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winSize,winSize),levels,iters,useInital);
        _parameters[1]->changed = false;
        _parameters[2]->changed = false;
        _parameters[3]->changed = false;
        _parameters[4]->changed = false;
    }
    std::vector<cv::cuda::GpuMat> grey_pyramid(*getParameter<int>(2)->Data() + 1);

    cv::cuda::GpuMat grey_img;
    if(img.channels() != 1)
        cv::cuda::cvtColor(img, grey_pyramid[0], cv::COLOR_BGR2GRAY,0, stream);
    else
        grey_pyramid[0] = img;
    build_pyramid(grey_pyramid, grey_pyramid.size(), stream);

    if(prev_grey.empty())
    {
        prev_grey = grey_pyramid;
        cv::cuda::GpuMat* inputPts = getParameter<cv::cuda::GpuMat>(0)->Data();
        if (inputPts && prev_key_points.empty())
        {
            inputPts->copyTo(prev_key_points, stream);
        }
        return img;
    }else
    {
        cv::cuda::GpuMat* inputPts = getParameter<cv::cuda::GpuMat>(0)->Data();
        if(!inputPts && prev_key_points.empty())
            return img;

        if(!inputPts->empty() || !prev_key_points.empty())
        {
            if(prev_key_points.empty())
                inputPts->copyTo(prev_key_points, stream);

            cv::cuda::GpuMat status, error, tracked_points;

            if (*getParameter<bool>(4)->Data() && tracked_points.empty())
                prev_key_points.copyTo(tracked_points, stream);
            optFlow->calc(prev_grey, grey_pyramid, prev_key_points, tracked_points, status, error, stream);

            updateParameter("Tracked points", tracked_points)->type = Parameters::Parameter::Output;
            updateParameter("Status", status)->type = Parameters::Parameter::Output;
            updateParameter("Error", error)->type = Parameters::Parameter::Output;
            
            prev_key_points = tracked_points;
            prev_grey = grey_pyramid;
        }
    }
    return img;
}


void BroxOpticalFlow::NodeInit(bool firstInit)
{

}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}


