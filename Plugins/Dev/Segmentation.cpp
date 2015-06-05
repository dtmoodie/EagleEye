#include "Segmentation.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudalegacy.hpp>

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudabgsegm")
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudalegacy")

using namespace EagleLib;
IPerModuleInterface* CALL GetModule()
{
    return PerModuleInterface::GetInstance();
}

void OtsuThreshold::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input Histogram", "Optional");
        addInputParameter<cv::Mat>("Input range", "Required if input histogram is provided");
    }

}

cv::cuda::GpuMat OtsuThreshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(img.channels() != 1)
    {
        log(Error, "Currently only support single channel images!");
        return img;
    }
    cv::cuda::GpuMat hist;
    cv::cuda::GpuMat* histogram = getParameter<cv::cuda::GpuMat*>(0)->data;
    cv::Mat* bins = getParameter<cv::Mat*>(1)->data;
    if(!histogram)
    {
        cv::Mat h_levels(1,200,CV_32F);
        double minVal, maxVal;
        stream.waitForCompletion();
        cv::cuda::minMax(img, &minVal, &maxVal);
        // Generate 300 equally spaced bins over the space
        double step = (maxVal - minVal) / double(200);

        double val = minVal;
        for(int i = 0; i < 200; ++i, val += step)
        {
            h_levels.at<float>(i) = val;
        }
        cv::cuda::histRange(img, hist, cv::cuda::GpuMat(h_levels), stream);
    }else
    {
        if(bins == nullptr)
        {
            log(Error, "Histogram provided but range not provided");
            return img;
        }
        if(bins->channels() != 1)
        {
            log(Error, "Currently only support equal bins accross all histograms");
            return img;
        }
        hist = *histogram;
    }

    // Normalize histogram
    hist.convertTo(hist, CV_32F, 1 / float(img.size().area()), 0, stream);

    // Download histogram
    cv::cuda::HostMem h_hist;
    hist.download(h_hist, stream);
    stream.waitForCompletion();
    cv::Mat h_hist_ = h_hist.createMatHeader();
    int channels = h_hist_.channels();
    std::vector<double> optValue(channels);


    if(channels == 1)
    {
        float prbn = 0;  // First order cumulative
        float meanItr = 0; // Second order cumulative
        float meanGlb = 0; // Global mean level
        float param1 = 0;
        float param2 = 0;
        double param3 = 0;
        double optThresh = 0;

        for(int i = 0; i < h_hist_.size().area(); ++i)
        {
            meanGlb += h_hist_.at<float>(i)*i;
        }



        // Currently we only support equal bins accross all channels
        float val = 0;
        for(int i = 0; i < bins->cols-1; ++i)
        {
            val = h_hist_.at<float>(i);
            prbn += val;
            meanItr += val * i;

            param1 = meanGlb * prbn - meanItr;
            param2 = param1 * param1 / (prbn*(1-prbn));
            if(param2 > param3)
            {
                param3 = param2;
                if(bins->type() == CV_32F)
                    optThresh = bins->at<float>(i);
                else
                    optThresh = bins->at<int>(i);
            }
        }
        optValue[0] = optThresh;
    }else
    {
        if(channels == 4)
        {
            for(int c = 0; c < channels; ++c)
            {
                float prbn = 0;  // First order cumulative
                float meanItr = 0; // Second order cumulative
                float meanGlb = 0; // Global mean level
                float param1 = 0;
                float param2 = 0;
                double param3 = 0;
                double optThresh = 0;

                for(int i = 0; i < h_hist_.size().area(); ++i)
                {
                    meanGlb += h_hist_.at<cv::Vec4f>(i).val[c]*i;
                }



                // Currently we only support equal bins accross all channels
                float val = 0;
                for(int i = 0; i < bins->size().area(); ++i)
                {
                    val = h_hist_.at<cv::Vec4f>(i).val[c];
                    prbn += val;
                    meanItr += val * i;

                    param1 = meanGlb * prbn - meanItr;
                    param2 = param1 * param1 / (prbn*(1-prbn));
                    if(param2 > param3)
                    {
                        param3 = param2;
                        optThresh = bins->at<float>(i);
                    }
                }
                optValue[c] = optThresh;
            }
        }else
        {
            log(Error, "Incompatible channel count");
        }
    }
    for(int i = 0; i < optValue.size(); ++i)
    {
        updateParameter("Optimal threshold " + boost::lexical_cast<std::string>(i), optValue[i], Parameter::Output);
    }
    return img;
}

void SegmentMOG2::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("History", int(500));
        updateParameter("Threshold", double(16));
        updateParameter("Detect Shadows", true);

    }
    updateParameter("Learning Rate", double(1.0));

}
void SegmentMOG2::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(mog2)
}

cv::cuda::GpuMat SegmentMOG2::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    //std::cout << "Test" << std::endl;
    if(parameters[0]->changed ||
        parameters[1]->changed ||
        parameters[2]->changed)
    {
        mog2 = cv::cuda::createBackgroundSubtractorMOG2(getParameter<int>(0)->data,getParameter<double>(1)->data, getParameter<bool>(2)->data);
    }
    if(mog2 != nullptr)
    {
        cv::cuda::GpuMat mask;
        mog2->apply(img, mask, getParameter<double>(3)->data, stream);
        updateParameter("Foreground mask", mask, Parameter::Output);
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

void SegmentCPMC::Init(bool firstInit)
{

}

cv::cuda::GpuMat SegmentCPMC::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}


void SegmentGrabCut::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::Mat>("Initial mask", "Optional");
        addInputParameter<cv::Rect>("ROI", "Optional");
        addInputParameter<cv::cuda::GpuMat>("Gpu initial mask");
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
    cv::Mat h_img, mask;
    img.download(h_img, stream);
    stream.waitForCompletion();
    cv::cuda::GpuMat* d_mask = getParameter<cv::cuda::GpuMat*>("Gpu initial mask")->data;
    cv::Mat* maskPtr = getParameter<cv::Mat*>(0)->data;
    int mode = getParameter<EnumParameter>(2)->data.getValue();
    bool maskExists = true;
    if(!maskPtr && !d_mask)
    {
        if(mode == cv::GC_INIT_WITH_MASK)
        {
            log(Error, "Mode set to initialize with mask, but no mask provided");
            return img;
        }
        maskExists = false;
    }
    if(maskPtr == nullptr && d_mask)
    {
        d_mask->download(mask, stream);
    }
    if(maskPtr)
    {
        mask = *maskPtr;
    }


    if(mode == cv::GC_INIT_WITH_MASK && mask.size() != h_img.size())
    {
        log(Error, "Mask size does not match image size");
        return img;
    }

    cv::Rect* roi = getParameter<cv::Rect*>(1)->data;
    if(!roi && mode == cv::GC_INIT_WITH_RECT)
    {
        log(Error, "Mode set to initialize with rect, but no rect provided");
        return img;
    }
    cv::Rect rect;
    if(roi == nullptr)
        roi = & rect;

    cv::grabCut(h_img, mask, *roi, bgdModel, fgdModel, getParameter<int>(3)->data, mode);
    if(!maskExists)
    {
        updateParameter("Grab Cut results", mask);
    }
    return img;
}

void KMeans::Init(bool firstInit)
{

}

cv::cuda::GpuMat KMeans::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}


void SegmentKMeans::Init(bool firstInit)
{
    EnumParameter flags;
    flags.addEnum(ENUM(cv::KMEANS_PP_CENTERS));
    flags.addEnum(ENUM(cv::KMEANS_RANDOM_CENTERS));
    flags.addEnum(ENUM(cv::KMEANS_USE_INITIAL_LABELS));
    updateParameter("K", int(10));
    updateParameter("Iterations", 100);
    updateParameter("Epsilon", double(0.1));
    updateParameter("Attempts", int(1));
    updateParameter("Flags", flags);
    updateParameter("Color weight", double(1.0));
    updateParameter("Distance weight", double(1.0));
}

cv::cuda::GpuMat SegmentKMeans::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    int k = getParameter<int>(0)->data;

    img.download(hostBuf, stream);
    stream.waitForCompletion();
    cv::Mat samples = hostBuf.createMatHeader();

    cv::Mat labels;
    cv::Mat clusters;
    cv::TermCriteria termCrit( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, getParameter<int>(1)->data, getParameter<double>(2)->data);
    double ret = cv::kmeans(samples, k, labels, termCrit, getParameter<int>(3)->data, getParameter<EnumParameter>(4)->data.getValue(), clusters);
    cv::cuda::GpuMat d_clusters, d_labels;
    d_clusters.upload(clusters, stream);
    d_labels.upload(labels, stream);
    updateParameter("Clusters", d_clusters, Parameter::Output);
    updateParameter("Labels", d_labels, Parameter::Output);
    updateParameter("Compactedness", ret, Parameter::Output);
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
    if(img.depth() != CV_8U)
    {
        log(Error, "Image not CV_8U type");
        return img;
    }
    if(img.channels() != 4)
    {
        log(Warning, "Image doesn't have 4 channels, appending blank image");
        if(blank.size() != img.size())
        {
            blank.create(img.size(), CV_8U);
            blank.setTo(cv::Scalar(0), stream);
        }
        std::vector<cv::cuda::GpuMat> channels;
        channels.push_back(img);
        channels.push_back(blank);
        cv::cuda::merge(channels, img, stream);
    }
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
    parameters[0]->changed = true;
}

cv::cuda::GpuMat ManualMask::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed || parameters.size() == 4)
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
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(OtsuThreshold)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentMOG2)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentGrabCut)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentWatershed)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentKMeans)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ManualMask)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentMeanShift)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentCPMC)
