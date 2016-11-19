#include "Binary.h"
#include "opencv2/imgproc.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;


SETUP_PROJECT_IMPL

bool MorphologyFilter::ProcessImpl()
{
    if (input_image)
    {
        if (structuring_element_type_param.modified || morphology_type_param.modified || 
            anchor_point_param.modified || iterations_param.modified ||
            filter == nullptr)
        {
            structuring_element_param.UpdateData(
                cv::getStructuringElement(
                    structuring_element_type.currentSelection,
                    ::cv::Size(structuring_element_size, structuring_element_size), anchor_point));

            filter = ::cv::cuda::createMorphologyFilter(
                morphology_type.currentSelection, input_image->GetMat(*_ctx->stream).type(), 
                structuring_element, anchor_point, iterations);

            structuring_element_size_param.modified = false;
            morphology_type_param.modified = false;
            anchor_point_param.modified = false;
            iterations_param.modified = false;
        }
        cv::cuda::GpuMat out;
        filter->apply(input_image->GetGpuMat(Stream()), out, Stream());
        this->output_param.UpdateData(out, input_image_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(MorphologyFilter);

/*
cv::cuda::GpuMat MorphologyFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    bool updateFilter = _parameters.size() != 7;
    if(_parameters[0]->changed || _parameters[2]->changed)
    {
        int size = *getParameter<int>(2)->Data();
        cv::Point anchor = *getParameter<cv::Point>(3)->Data();
        updateParameter(4, cv::getStructuringElement(
                            getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection,
                            cv::Size(size,size),anchor));

        updateFilter = true;
        _parameters[0]->changed = false; 
        _parameters[2]->changed = false;
        //log(Status,"Structuring element updated");
        NODE_LOG(info) << "Structuring element updated";
    }
    if(_parameters[1]->changed || updateFilter)
    {
        updateParameter("Filter",
            cv::cuda::createMorphologyFilter(
                getParameter<Parameters::EnumParameter>(1)->Data()->currentSelection,img.type(),
                *getParameter<cv::Mat>(4)->Data(),
                *getParameter<cv::Point>(3)->Data(),
                *getParameter<int>(5)->Data()));
        NODE_LOG(info) << "Filter updated";
        _parameters[1]->changed = false; 
    }
    cv::cuda::GpuMat output;
    (*getParameter<cv::Ptr<cv::cuda::Filter>>(6)->Data())->apply(img,output,stream);
    return output;
}
*/


bool FindContours::ProcessImpl()
{
    if(input_image)
    {
        ::cv::Mat h_mat = input_image->GetMat(*_ctx->stream);
        if(::cv::countNonZero(h_mat) <= 1)
        {
            this->contours.clear();
            num_contours = 0;
            return false;
        }
        long long ts = input_image_param.GetTimestamp();
        _ctx->stream->waitForCompletion();
        ::cv::findContours(h_mat,contours, hierarchy, mode.currentSelection, method.currentSelection);
        contours_param.Commit(ts, _ctx);
        hierarchy_param.Commit(ts, _ctx);
        num_contours_param.UpdateData(contours.size(), ts, _ctx);
        return true;
    }
    return false;
}

/*TS<SyncedMemory> FindContours::doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream)
{
    cv::Mat h_mat = img.GetMat(stream).clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    if(cv::countNonZero(h_mat) <= 1)
    {
        getParameter<std::vector<cv::Vec4i>>(3)->Data()->clear();
        return img;
    }
    cv::findContours(h_mat,
        contours,
        *getParameter<std::vector<cv::Vec4i>>(3)->Data(),
        getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection,
        getParameter<Parameters::EnumParameter>(1)->Data()->currentSelection);
    
    updateParameter<int>("Contours found", contours.size())->type =  Parameters::Parameter::State;
    updateParameter("Contours", contours)->type = Parameters::Parameter::Output;
    updateParameter("Hierarchy", hierarchy)->type = Parameters::Parameter::Output;
    if (*getParameter<bool>(4)->Data())
    {
        if (_parameters[4]->changed)
        {
            updateParameter<bool>("Oriented Area", false);
            updateParameter<bool>("Filter area", false);
            _parameters[4]->changed = false;
        }
        auto areaParam = getParameter<bool>("Filter area");
        if (areaParam != nullptr && *areaParam->Data() && areaParam->changed)
        {
            updateParameter<double>("Filter threshold", 0.0);
            updateParameter<double>("Filter sigma", 0.0);
            areaParam->changed = false;
        }
        auto areaPtr = getParameter<std::vector<std::pair<int, double>>>("Contour Area")->Data();
        bool oriented = *getParameter<bool>("Oriented Area")->Data();
        areaPtr->resize(contours.size());
        for (size_t i = 0; i < contours.size(); ++i)
        {
            (*areaPtr)[i] = std::pair<int, double>(int(i), cv::contourArea(contours[i], oriented));
        }
        auto thresholdParam = getParameter<double>("Filter threshold");
        if (thresholdParam != nullptr && thresholdParam->Data() != nullptr)
        {
            areaPtr->erase(std::remove_if(areaPtr->begin(), areaPtr->end(),
                [thresholdParam](std::pair<int, double> x) {return x.second < *thresholdParam->Data(); }), areaPtr->end());
        }
        auto sigmaParam = getParameter<double>("Filter sigma");
        if (sigmaParam != nullptr && *sigmaParam->Data() != 0.0)
        {
            // Calculate mean and sigma
            double sum = 0;
            double sumSq = 0;
            for (size_t i = 0; i < areaPtr->size(); ++i)
            {
                sum += (*areaPtr)[i].second;
                sumSq += (*areaPtr)[i].second*(*areaPtr)[i].second;
            }
        }
    }
    return img;
}*/


/*void ContourBoundingBox::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<std::vector<std::vector<cv::Point>>>("Contours");
        addInputParameter<std::vector<cv::Vec4i>>("Hierarchy");
        ParameteredObject::addParameter<cv::Scalar>("Box color", cv::Scalar(0,0,255));
        ParameteredObject::addParameter<int>("Line thickness", 2);
        addInputParameter<std::vector<std::pair<int,double>>>("Contour Area");
        updateParameter<bool>("Use filtered area", false);
    }
    updateParameter<bool>("Merge contours", false);

}*/
bool ContourBoundingBox::ProcessImpl()
{
    if(this->contours && this->input_image)
    {
        std::vector<cv::Rect> boxes;
        for(size_t i = 0; i < contours->size(); ++i)
        {
            boxes.push_back(cv::boundingRect((*contours)[i]));   
        }
        if(merge_contours)
        {
            for(size_t i = 0; i < boxes.size(); ++i)
            {
            
            }
        }
    }
    return false;
}
/*TS<SyncedMemory> ContourBoundingBox::doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream)
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
        updateParameter<int>("Separation distance", 5)->SetTooltip("Max distance between contours to still merge contours");
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

    cv::Mat h_img = img.GetMat(stream);
    stream.waitForCompletion();
    h_img = h_img.clone();
    cv::Scalar replace;
    if(h_img.channels() == 3)
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
            cv::rectangle(h_img, boxes[(*areaParam->Data())[i].first].tl(), boxes[(*areaParam->Data())[i].first].br(), replace, lineWidth);
        }
    }else
    {
        for(size_t i = 0; i < boxes.size(); ++i)
        {
            cv::rectangle(h_img, boxes[i].tl(), boxes[i].br(),replace, lineWidth);
        }
    }
    
    return TS<SyncedMemory>(img.timestamp, img.frame_number, h_img);
}*/


/*void HistogramThreshold::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        Parameters::EnumParameter param;
        param.addEnum(ENUM(KeepCenter));
        param.addEnum(ENUM(SuppressCenter));
        updateParameter("Threshold type", param);
        updateParameter("Threshold width", 0.5)->SetTooltip("Percent of histogram to threshold");
        addInputParameter<cv::cuda::GpuMat>("Input histogram");
        addInputParameter<cv::cuda::GpuMat>("Input image");
        addInputParameter<cv::cuda::GpuMat>("Input mask");
        addInputParameter<cv::Mat>("Histogram bins");
    }
}*/

/*void histogramThresholdCallback(int status, void* userData)
{
    HistogramThreshold* node = (HistogramThreshold*)userData;
    node->runFilter();
}*/


void HistogramThreshold::runFilter()
{

}

/*cv::cuda::GpuMat HistogramThreshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
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
        NODE_LOG(error) << "Image to threshold needs to be a single channel image";
        return img;
    }
    cv::cuda::HostMem histogram;
    inputHistogram->download(histogram, stream);
    stream.waitForCompletion();
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
    updateParameter("Threshold min value", thresholdMin)->type = Parameters::Parameter::Output;
    updateParameter("Threshold max value", thresholdMax)->type =  Parameters::Parameter::Output;
    updateParameter("Max Idx", maxIdx);
    cv::cuda::GpuMat output;
    switch(type)
    {
    case KeepCenter:
        // We want to threshold such that just the center passes
        // To do this, we threshold a positive mask for all values greater than the min
        // As well as for all values below the max, then we AND them together.
        cv::cuda::threshold(img, lowerMask, thresholdMin, 255, cv::THRESH_BINARY, stream);
        cv::cuda::threshold(img, upperMask, thresholdMax, 255, cv::THRESH_BINARY_INV, stream);
        
        cv::cuda::bitwise_and(lowerMask, upperMask, output, cv::noArray(), stream);
        return output;
    case SuppressCenter:
        cv::cuda::threshold(img, lowerMask, thresholdMax, 255, cv::THRESH_BINARY, stream);
        cv::cuda::threshold(img, upperMask, thresholdMin, 255, cv::THRESH_BINARY_INV, stream);
        cv::cuda::bitwise_or(lowerMask, upperMask, output, cv::noArray(), stream);
    }
    updateParameter("Image mask", output)->type =  Parameters::Parameter::Output;
    return output;
}*/

/*void PruneContours::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<std::vector<std::vector<cv::Point>>>("Input Contours");
    }
}*/

/*TS<SyncedMemory> PruneContours::doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream)
{
    auto input = getParameter<std::vector<std::vector<cv::Point>>>("Input Contours")->Data();
    if(input)
    {
        std::vector<std::vector<cv::Point>> output;
        for(auto& contour : *input)
        {
            if((contour.size() > min_area || min_area == -1) && (contour.size() < max_area || max_area == -1))
            {
                output.push_back(contour);
            }
        }
        updateParameter("Pruned Contours", output)->type = Parameters::Parameter::Output;
        updateParameter("Num Pruned Contours", output.size())->type = Parameters::Parameter::State;
    }
    return img;
}*/

/*void DrawContours::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<std::vector<std::vector<cv::Point>>>("Input Contours");    
    }    
}*/

/*TS<SyncedMemory> DrawContours::doProcess(TS<SyncedMemory> img, cv::cuda::Stream& stream)
{
    auto input = getParameter<std::vector<std::vector<cv::Point>>>("Input Contours")->Data();
    if(input)
    {
        cv::Mat h_mat = img.GetMat(stream);
        //cv::drawContours(h_mat, *input, -1, 
    }
    return img;
}*/

/*void DrawRects::NodeInit(bool firstInit)
{
}
cv::cuda::GpuMat DrawRects::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}*/


//NODE_DEFAULT_CONSTRUCTOR_IMPL(PruneContours, Contours);
//NODE_DEFAULT_CONSTRUCTOR_IMPL(MorphologyFilter, Image, Processing)
//NODE_DEFAULT_CONSTRUCTOR_IMPL(FindContours, Image, Extractor)
//NODE_DEFAULT_CONSTRUCTOR_IMPL(ContourBoundingBox, Image, Processing)
//NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramThreshold, Image, Processing)
