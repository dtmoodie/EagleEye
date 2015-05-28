#include "Segmentation.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace EagleLib;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}

void OtsuThreshold::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input Histogram", "Optional");
    }
}

cv::cuda::GpuMat OtsuThreshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    cv::cuda::GpuMat* histogram = getParameter<cv::cuda::GpuMat*>(0)->data;
    if(!histogram)
    {
        // Calculate histogram here
    }

    return img;
}


void SegmentWatershed::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::Mat>("Input Marker Mask");
    }
}

cv::cuda::GpuMat SegmentWatershed::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    cv::Mat h_img;
    img.download(h_img,stream);
    cv::Mat* h_markerMask = getParameter<cv::Mat*>(0)->data;
    if(h_markerMask)
    {
        cv::watershed(h_img, *h_markerMask);
    }

    return img;
}

void SegmentGrabCut::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::Mat>("Initial mask", "Optional");
        addInputParameter<cv::Rect>("ROI", "Optional");
        EnumParameter param;
        param.addEnum(ENUM(cv::GC_INIT_WITH_RECT));
        param.addEnum(ENUM(cv::GC_INIT_WITH_MASK));
        param.addEnum(ENUM(cv::GC_EVAL));
        updateParameter("Grabcut mode", param);
        updateParameter("Iterations", int(10));

    }
    maskBuf.resize(5);
}

cv::cuda::GpuMat SegmentGrabCut::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    cv::Mat h_img;
    img.download(h_img, stream);
    stream.waitForCompletion();
    cv::Mat* mask = getParameter<cv::Mat*>(0)->data;
    int mode = getParameter<EnumParameter>(2)->data.getValue();
    bool maskExists = true;
    if(!mask)
    {
        if(mode == cv::GC_INIT_WITH_MASK)
        {
            log(Error, "Mode set to initialize with mask, but no mask provided");
            return img;
        }
        maskExists = false;
        mask = maskBuf.getFront();
    }

    cv::Rect* roi = getParameter<cv::Rect*>(1)->data;
    if(!roi && mode == cv::GC_INIT_WITH_RECT)
    {
        log(Error, "Mode set to initialize with rect, but no rect provided");
        return img;
    }
    cv::grabCut(h_img,*mask, *roi, bgdModel, fgdModel, getParameter<int>(3)->data, mode);
    if(!maskExists)
    {
        updateParameter("Grab Cut results", *mask);
    }
    return img;
}
void SegmentKMeans::Init(bool firstInit)
{

}

cv::cuda::GpuMat SegmentKMeans::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}
void
SegmentMeanShift::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Spatial window radius", int(5));
        updateParameter("Color radius", int(5));
        updateParameter("Min size", int(5));
        updateParameter("Max iterations", 5);
        updateParameter("Epsilon", double(1.0));
    }
}

cv::cuda::GpuMat SegmentMeanShift::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    cv::cuda::meanShiftSegmentation(img, dest,
        getParameter<int>(0)->data,
        getParameter<int>(1)->data,
        getParameter<int>(2)->data,
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, getParameter<int>(3)->data,
        getParameter<double>(4)->data), stream);
    img.upload(dest,stream);
    return img;
}



void ManualMask::Init(bool firstInit)
{

    if(firstInit)
    {
        EnumParameter param;
        param.addEnum(ENUM(Circular));
        param.addEnum(ENUM(Rectangular));
        updateParameter("Type", param);
        updateParameter("Origin", cv::Scalar(0,0));
        updateParameter("Size", cv::Scalar(5,5));
        updateParameter("Radius", int(5));
        updateParameter("Inverted", false);
    }
}

cv::cuda::GpuMat ManualMask::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed)
    {
        bool inverted = getParameter<bool>(4)->data;
        cv::Scalar origin = getParameter<cv::Scalar>(1)->data;
        cv::Mat h_mask;
        if(inverted)
            h_mask = cv::Mat(img.size(), CV_8U, cv::Scalar(0));
        else
            h_mask = cv::Mat(img.size(), CV_8U, cv::Scalar(255));

        switch(getParameter<EnumParameter>(0)->data.getValue())
        {

        case Circular:

            if(inverted)
                cv::circle(h_mask, cv::Point(origin.val[0], origin.val[1]), getParameter<int>(3)->data, cv::Scalar(255), -1);
            else
                cv::circle(h_mask, cv::Point(origin.val[0], origin.val[1]), getParameter<int>(3)->data, cv::Scalar(0), -1);
            break;
        case Rectangular:
            cv::Scalar size = getParameter<cv::Scalar>(2)->data;
            if(inverted)
                cv::rectangle(h_mask, cv::Rect(origin.val[0], origin.val[1], size.val[0], size.val[1]), cv::Scalar(255),-1);
            else
                cv::rectangle(h_mask, cv::Rect(origin.val[0], origin.val[1], size.val[0], size.val[1]), cv::Scalar(0),-1);
        }
        updateParameter("Manually defined mask", cv::cuda::GpuMat(h_mask), Parameter::Output);
    }
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(OtsuThreshold)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentGrabCut)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentWatershed)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentKMeans)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ManualMask)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentMeanShift)
