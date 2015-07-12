#include "nodes/ImgProc/Binary.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <algorithm>
#include <utility>
using namespace EagleLib;

void MorphologyFilter::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		Parameters::EnumParameter structuringElement;
        structuringElement.addEnum(ENUM(cv::MORPH_RECT));
        structuringElement.addEnum(ENUM(cv::MORPH_CROSS));
        structuringElement.addEnum(ENUM(cv::MORPH_ELLIPSE));
        updateParameter("Structuring Element Type", structuringElement);    // 0
		Parameters::EnumParameter morphType;
        morphType.addEnum(ENUM(cv::MORPH_ERODE));
        morphType.addEnum(ENUM(cv::MORPH_DILATE));
        morphType.addEnum(ENUM(cv::MORPH_OPEN));
        morphType.addEnum(ENUM(cv::MORPH_CLOSE));
        morphType.addEnum(ENUM(cv::MORPH_GRADIENT));
        morphType.addEnum(ENUM(cv::MORPH_TOPHAT));
        morphType.addEnum(ENUM(cv::MORPH_BLACKHAT));
        updateParameter("Morphology type", morphType);  //1
        updateParameter("Structuring Element Size", int(5)); // 2
        updateParameter("Anchor Point", cv::Point(-1,-1));  // 3
        updateParameter("Structuring Element", cv::getStructuringElement(0,cv::Size(5,5))); // 4
        updateParameter("Iterations", int(1));
    }
}

cv::cuda::GpuMat MorphologyFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    bool updateFilter = parameters.size() != 7;
    if(parameters[0]->changed || parameters[2]->changed)
    {
        int size = *getParameter<int>(2)->Data();
        cv::Point anchor = *getParameter<cv::Point>(3)->Data();
        updateParameter(4, cv::getStructuringElement(getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection,
                                  cv::Size(size,size),anchor));

        updateFilter = true;
        parameters[0]->changed = false;
        parameters[2]->changed = false;
        log(Status,"Structuring element updated");
    }
    if(parameters[1]->changed || updateFilter)
    {
        updateParameter("Filter",
            cv::cuda::createMorphologyFilter(
                getParameter<Parameters::EnumParameter>(1)->Data()->currentSelection,img.type(),
                *getParameter<cv::Mat>(4)->Data(),
                *getParameter<cv::Point>(3)->Data(),
                *getParameter<int>(5)->Data()));
        log(Status, "Filter updated");
        parameters[1]->changed = false;
    }
    (*getParameter<cv::Ptr<cv::cuda::Filter>>(6)->Data())->apply(img,img,stream);
    return img;
}
void FindContours::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		Parameters::EnumParameter mode;
        mode.addEnum(ENUM(cv::RETR_EXTERNAL));
        mode.addEnum(ENUM(cv::RETR_LIST));
        mode.addEnum(ENUM(cv::RETR_CCOMP));
        mode.addEnum(ENUM(cv::RETR_TREE));
        mode.addEnum(ENUM(cv::RETR_FLOODFILL));
		Parameters::EnumParameter method;
        method.addEnum(ENUM(cv::CHAIN_APPROX_NONE));
        method.addEnum(ENUM(cv::CHAIN_APPROX_SIMPLE));
        method.addEnum(ENUM(cv::CHAIN_APPROX_TC89_L1));
        method.addEnum(ENUM(cv::CHAIN_APPROX_TC89_KCOS));
        updateParameter("Mode", mode);      // 0
        updateParameter("Method", method);  // 1
        updateParameter<std::vector<std::vector<cv::Point>>>("Contours", std::vector<std::vector<cv::Point>>(), Parameters::Parameter::Output); // 2
        updateParameter<std::vector<cv::Vec4i>>("Hierarchy", std::vector<cv::Vec4i>()); // 3
        updateParameter<bool>("Calculate contour Area", false); // 4
        updateParameter<bool>("Calculate Moments", false);  // 5
    }

}

cv::cuda::GpuMat FindContours::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::Mat h_img;
    img.download(h_img, stream);
    stream.waitForCompletion();
    std::vector<std::vector<cv::Point> >* ptr = getParameter<std::vector<std::vector<cv::Point>>>(2)->Data();
    cv::findContours(h_img,
        *ptr,
        *getParameter<std::vector<cv::Vec4i>>(3)->Data(),
        getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection,
        getParameter<Parameters::EnumParameter>(1)->Data()->currentSelection);
    updateParameter<int>("Contours found",ptr->size(), Parameters::Parameter::State);
    parameters[2]->changed = true;
    parameters[3]->changed = true;
    if(*getParameter<bool>(4)->Data())
    {
        if(parameters[4]->changed)
        {
            updateParameter<std::vector<std::pair<int,double>>>("Contour Area",std::vector<std::pair<int,double>>(), Parameters::Parameter::Output);
            updateParameter<bool>("Oriented Area",false);
            updateParameter<bool>("Filter area", false);
            parameters[4]->changed = false;
        }
        auto areaParam = getParameter<bool>("Filter area");
        if(areaParam != nullptr && *areaParam->Data() && areaParam->changed)
        {
            updateParameter<double>("Filter threshold", 0.0);
            updateParameter<double>("Filter sigma", 0.0);
            areaParam->changed = false;
        }
        auto areaPtr = getParameter<std::vector<std::pair<int,double>>>("Contour Area")->Data();
        bool oriented = *getParameter<bool>("Oriented Area")->Data();
        areaPtr->resize(ptr->size());
        for(size_t i = 0; i < ptr->size(); ++i)
        {
            (*areaPtr)[i] = std::pair<int,double>(int(i),cv::contourArea((*ptr)[i], oriented));
        }
        auto thresholdParam = getParameter<double>("Filter threshold");
        if(thresholdParam != nullptr && thresholdParam->Data() != nullptr)
        {
            areaPtr->erase(std::remove_if(areaPtr->begin(), areaPtr->end(),
                            [thresholdParam](std::pair<int,double> x){return x.second < *thresholdParam->Data();}), areaPtr->end());
            // This should be more efficient, needs to be tested though
            /*for(auto it = areaPtr->begin(); it != areaPtr->end(); ++it)
            {
                if(it->second < thresholdParam->data)
                {
                    std::swap(*it, areaPtr->back());
                    areaPtr->pop_back();
                }
            }*/
        }
        auto sigmaParam = getParameter<double>("Filter sigma");
        if(sigmaParam != nullptr && *sigmaParam->Data() != 0.0)
        {
            // Calculate mean and sigma
            double sum = 0;
            double sumSq = 0;
            for(size_t i = 0; i < areaPtr->size(); ++i)
            {
                sum += (*areaPtr)[i].second;
                sumSq += (*areaPtr)[i].second*(*areaPtr)[i].second;
            }

        }
    }

    return img;
}
void ContourBoundingBox::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<std::vector<std::vector<cv::Point>>>("Contours");
        addInputParameter<std::vector<cv::Vec4i>>("Hierarchy");
        addParameter<cv::Scalar>("Box color", cv::Scalar(0,0,255));
        addParameter<int>("Line thickness", 2);
        addInputParameter<std::vector<std::pair<int,double>>>("Contour Area");
        updateParameter<bool>("Use filtered area", false);
    }
    updateParameter<bool>("Merge contours", false);

}

cv::cuda::GpuMat ContourBoundingBox::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto contourPtr = getParameter<std::vector<std::vector<cv::Point>>>(0)->Data();
    if(!contourPtr)
        return img;
    std::vector<cv::Rect> boxes;
    for(size_t i = 0; i < contourPtr->size(); ++i)
    {
        boxes.push_back(cv::boundingRect((*contourPtr)[i]));
    }
    auto mergeParam = getParameter<bool>("Merge contours");
    if(mergeParam && mergeParam->changed)
    {
        updateParameter<int>("Separation distance", 5, Parameters::Parameter::Control, "Max distance between contours to still merge contours");
    }
    if(mergeParam && *mergeParam->Data())
    {
        //int distance = getParameter<int>("Separation distance")->data;
        for(size_t i = 0; i < boxes.size() - 1; ++i)
        {
            for(size_t j = i + 1; j < boxes.size(); ++j)
            {
                // Check distance between bounding rects
                cv::Point c1 = boxes[i].tl() + cv::Point(boxes[i].width/2, boxes[i].height/2);
                cv::Point c2 = boxes[j].tl() + cv::Point(boxes[j].width/2, boxes[j].height/2);
                auto dist = cv::norm(c1 - c2);
                auto thresh = 1.3*(cv::norm(boxes[i].tl() - c1) + cv::norm(boxes[j].tl() - c2));
                if(dist > thresh)
                    continue;

                // If we've made it this far, then we need to merge the rectangles
                cv::Rect newRect = boxes[i] | boxes[j];
                boxes[i] = newRect;
                boxes.erase(boxes.begin() + j);
            }
        }
    }

    cv::Mat h_img;
    img.download(h_img,stream);
    stream.waitForCompletion();
    cv::Scalar replace;
    if(img.channels() == 3)
        replace = *getParameter<cv::Scalar>(2)->Data();
    else
        replace = cv::Scalar(128,0,0);
    auto useArea = getParameter<bool>("Use filtered area");
    int lineWidth = *getParameter<int>(3)->Data();
    auto areaParam = getParameter<std::vector<std::pair<int,double>>>("Contour Area");
    if(useArea && *useArea->Data() && areaParam && areaParam->Data())
    {
        for(size_t i = 0; i < areaParam->Data()->size(); ++i)
        {
            cv::rectangle(h_img, boxes[(*areaParam->Data())[i].first], replace, lineWidth);
        }
    }else
    {
        for(size_t i = 0; i < boxes.size(); ++i)
        {
            cv::rectangle(h_img, boxes[i],replace, lineWidth);
        }
    }
    img.upload(h_img,stream);
    return img;
}
void HistogramThreshold::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		Parameters::EnumParameter param;
        param.addEnum(ENUM(KeepCenter));
        param.addEnum(ENUM(SuppressCenter));
        updateParameter("Threshold type", param);
        updateParameter("Threshold width", 0.5, Parameters::Parameter::Control, "Percent of histogram to threshold");
        addInputParameter<cv::cuda::GpuMat>("Input histogram");
        addInputParameter<cv::cuda::GpuMat>("Input image", "Optional");
        addInputParameter<cv::cuda::GpuMat>("Input mask", "Optional");
        addInputParameter<cv::Mat>("Histogram bins");
    }
}

void histogramThresholdCallback(int status, void* userData)
{
    HistogramThreshold* node = (HistogramThreshold*)userData;
    node->runFilter();
}


void HistogramThreshold::runFilter()
{
//    cv::Mat loc = currentLocBuffer->createMatHeader();
//    std::cout << loc.row(0) << std::endl;
//    switch(type)
//    {
//    case KeepCenter:
//        //

//        break;
//    case SuppressCenter:

//        break;
//    }
}

cv::cuda::GpuMat HistogramThreshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    inputHistogram = getParameter<cv::cuda::GpuMat>(2)->Data();
    inputImage = getParameter<cv::cuda::GpuMat>(3)->Data();
    inputMask = getParameter<cv::cuda::GpuMat>(4)->Data();
    type = (ThresholdType)getParameter<Parameters::EnumParameter>(0)->Data()->getValue();
    cv::Mat* bins = getParameter<cv::Mat>("Histogram bins")->Data();
    _stream = stream;
    if(bins == nullptr)
        return img;
    if(inputImage == nullptr)
        inputImage = &img;
    if(inputHistogram == nullptr)
        return img;
    if(img.channels() != 1)
    {
        log(Error, "Image to threshold needs to be a single channel image");
        return img;
    }
    cv::cuda::HostMem histogram;
    inputHistogram->download(histogram, stream);
    //cv::cuda::findMinMaxLoc(*inputHistogram, values, location, inputMask == nullptr ? cv::noArray(): *inputMask, stream);
    stream.waitForCompletion();
    //ss << values.createMatHeader().row(0) << " " << location.createMatHeader().row(0) << " " << histogram.createMatHeader().row(0) << std::endl;
    cv::Mat hist = histogram.createMatHeader();
    int maxVal = 0;
    int maxIdx = 0;
    for(int i = 0 ; i < hist.cols; ++i)
    {
        if(hist.at<int>(i) > maxVal)
        {
            maxVal = hist.at<int>(i);
            maxIdx = i;
        }
    }
    int numBins = hist.cols;
    int thresholdWidth = numBins * (*getParameter<double>(1)->Data())*0.5;
    int minBin = maxIdx - thresholdWidth;
    int maxBin = maxIdx + thresholdWidth;
    minBin = std::max(minBin, 0);
    maxBin = std::min(maxBin, hist.cols - 1);
    float thresholdMin = bins->at<float>(minBin);
    float thresholdMax = bins->at<float>(maxBin);
    updateParameter("Threshold min value", thresholdMin, Parameters::Parameter::Output);
	updateParameter("Threshold max value", thresholdMax, Parameters::Parameter::Output);
    updateParameter("Max Idx", maxIdx);
    switch(type)
    {
    case KeepCenter:
        // We want to threshold such that just the center passes
        // To do this, we threshold a positive mask for all values greater than the min
        // As well as for all values below the max, then we AND them together.
        cv::cuda::threshold(img, lowerMask, thresholdMin, 255, cv::THRESH_BINARY, stream);
        cv::cuda::threshold(img, upperMask, thresholdMax, 255, cv::THRESH_BINARY_INV, stream);
        cv::cuda::bitwise_and(lowerMask, upperMask, img, cv::noArray(), stream);
        break;
    case SuppressCenter:
        cv::cuda::threshold(img, lowerMask, thresholdMax, 255, cv::THRESH_BINARY, stream);
        cv::cuda::threshold(img, upperMask, thresholdMin, 255, cv::THRESH_BINARY_INV, stream);
        cv::cuda::bitwise_or(lowerMask, upperMask, img, cv::noArray(), stream);
    }
	updateParameter("Image mask", img, Parameters::Parameter::Output);

    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(MorphologyFilter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FindContours)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ContourBoundingBox)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramThreshold)
