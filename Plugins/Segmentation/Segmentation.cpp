#include "Segmentation.h"
#include <EagleLib/rcc/external_includes/cv_imgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudalegacy.hpp>
#include "RuntimeLinkLibrary.h"
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("fastmsd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("fastms.lib")
#endif
using namespace EagleLib;
using namespace EagleLib::Nodes;
SETUP_PROJECT_IMPL

void OtsuThreshold::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input Histogram")->SetTooltip("Optional");
        addInputParameter<cv::Mat>("Input range")->SetTooltip("Required if input histogram is provided");
    }

}

cv::cuda::GpuMat OtsuThreshold::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(img.channels() != 1)
    {
        //log(Error, "Currently only support single channel images!");
		NODE_LOG(warning) << "Currently only support single channel images!";
        return img;
    }
    cv::cuda::GpuMat hist;
    cv::cuda::GpuMat* histogram = getParameter<cv::cuda::GpuMat>(0)->Data();
    cv::Mat* bins = getParameter<cv::Mat>(1)->Data();
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
            //log(Error, "Histogram provided but range not provided");
			NODE_LOG(error) << "Histogram provided but range not provided";
            return img;
        }
        if(bins->channels() != 1)
        {
            //log(Error, "Currently only support equal bins accross all histograms");
			NODE_LOG(error) << "Currently only support equal bins accross all histograms";
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
            //log(Error, "Incompatible channel count");
			NODE_LOG(error) << "Incompatible channel count";
        }
    }
    for(int i = 0; i < optValue.size(); ++i)
    {
        updateParameter("Optimal threshold " + boost::lexical_cast<std::string>(i), optValue[i])->type =  Parameters::Parameter::Output;
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
        mog2 = cv::cuda::createBackgroundSubtractorMOG2(*getParameter<int>(0)->Data(),*getParameter<double>(1)->Data(), *getParameter<bool>(2)->Data());
    }
    if(mog2 != nullptr)
    {
        cv::cuda::GpuMat mask;
        mog2->apply(img, mask, *getParameter<double>(3)->Data(), stream);
        updateParameter("Foreground mask", mask)->type =  Parameters::Parameter::Output;
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
    cv::Mat* h_markerMask = getParameter<cv::Mat>(0)->Data();
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
        addInputParameter<cv::Mat>("Initial mask")->SetTooltip("Optional");
        addInputParameter<cv::Rect>("ROI")->SetTooltip("Optional");
        addInputParameter<cv::cuda::GpuMat>("Gpu initial mask");
        Parameters::EnumParameter param;
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
    cv::cuda::GpuMat* d_mask = getParameter<cv::cuda::GpuMat>("Gpu initial mask")->Data();
    cv::Mat* maskPtr = getParameter<cv::Mat>(0)->Data();
    int mode = getParameter<Parameters::EnumParameter>(2)->Data()->getValue();
    bool maskExists = true;
    if(!maskPtr && !d_mask)
    {
        if(mode == cv::GC_INIT_WITH_MASK)
        {
            //log(Error, "Mode set to initialize with mask, but no mask provided");
			NODE_LOG(error) << "Mode set to initialize with mask, but no mask provided";
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
        //log(Error, "Mask size does not match image size");
		NODE_LOG(error) << "Mask size does not match image size";
        return img;
    }

    cv::Rect* roi = getParameter<cv::Rect>(1)->Data();
    if(!roi && mode == cv::GC_INIT_WITH_RECT)
    {
        //log(Error, "Mode set to initialize with rect, but no rect provided");
		NODE_LOG(error) << "Mode set to initialize with rect, but no rect provided";
        return img;
    }
    cv::Rect rect;
    if(roi == nullptr)
        roi = & rect;

    cv::grabCut(h_img, mask, *roi, bgdModel, fgdModel, *getParameter<int>(3)->Data(), mode);
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
    Parameters::EnumParameter flags;
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
    int k = *getParameter<int>(0)->Data();

    img.download(hostBuf, stream);
    stream.waitForCompletion();
    cv::Mat samples = hostBuf.createMatHeader();

    cv::Mat labels;
    cv::Mat clusters;
    cv::TermCriteria termCrit( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, *getParameter<int>(1)->Data(), *getParameter<double>(2)->Data());
    double ret = cv::kmeans(samples, k, labels, termCrit, *getParameter<int>(3)->Data(), getParameter<Parameters::EnumParameter>(4)->Data()->getValue(), clusters);
    cv::cuda::GpuMat d_clusters, d_labels;
    d_clusters.upload(clusters, stream);
    d_labels.upload(labels, stream);
    updateParameter("Clusters", d_clusters)->type = Parameters::Parameter::Output;
    updateParameter("Labels", d_labels)->type = Parameters::Parameter::Output;
    updateParameter("Compactedness", ret)->type =  Parameters::Parameter::Output;
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
        //log(Error, "Image not CV_8U type");
		NODE_LOG(error) << "Image not CV_8U type";
        return img;
    }
    if(img.channels() != 4)
    {
        //log(Warning, "Image doesn't have 4 channels, appending blank image");
		NODE_LOG(warning) << "Image doesn't have 4 channels, appending blank image";
        if(blank.size() != img.size())
        {
            blank.create(img.size(), CV_8U);
            blank.setTo(cv::Scalar(0), stream);
        }
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(img,channels, stream);
        channels.push_back(blank);
        cv::cuda::merge(channels, img, stream);
    }
    cv::cuda::meanShiftSegmentation(img, dest,
        *getParameter<int>(0)->Data(),
        *getParameter<int>(1)->Data(),
        *getParameter<int>(2)->Data(),
        cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, *getParameter<int>(3)->Data(),
        *getParameter<double>(4)->Data()), stream);
    img.upload(dest,stream);
    return img;
}



void ManualMask::Init(bool firstInit)
{

    if(firstInit)
    {
        Parameters::EnumParameter param;
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
        bool inverted = *getParameter<bool>(4)->Data();
        cv::Scalar origin = *getParameter<cv::Scalar>(1)->Data();
        cv::Mat h_mask;
        if(inverted)
            h_mask = cv::Mat(img.size(), CV_8U, cv::Scalar(0));
        else
            h_mask = cv::Mat(img.size(), CV_8U, cv::Scalar(255));

        switch(getParameter<Parameters::EnumParameter>(0)->Data()->getValue())
        {

        case Circular:

            if(inverted)
                cv::circle(h_mask, cv::Point(origin.val[0], origin.val[1]), *getParameter<int>(3)->Data(), cv::Scalar(255), -1);
            else
                cv::circle(h_mask, cv::Point(origin.val[0], origin.val[1]), *getParameter<int>(3)->Data(), cv::Scalar(0), -1);
            break;
        case Rectangular:
            cv::Scalar size = *getParameter<cv::Scalar>(2)->Data();
            if(inverted)
                cv::rectangle(h_mask, cv::Rect(origin.val[0], origin.val[1], size.val[0], size.val[1]), cv::Scalar(255),-1);
            else
                cv::rectangle(h_mask, cv::Rect(origin.val[0], origin.val[1], size.val[0], size.val[1]), cv::Scalar(0),-1);
        }
        updateParameter("Manually defined mask", cv::cuda::GpuMat(h_mask))->type = Parameters::Parameter::Output;
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    return img;
}

void SLaT::Init(bool firstInit)
{
	Node::Init(firstInit);
	updateParameter("Lambda", double(0.1))->SetTooltip( "For bigger values, number of discontinuities will be smaller, for smaller values more discontinuities");
	updateParameter("Alpha", double(20.0))->SetTooltip("For bigger values, solution will be more flat, for smaller values, solution will be more rough.");
	updateParameter("Temporal", double(0.0))->SetTooltip("For bigger values, solution will be driven to be similar to the previous frame, smaller values will allow for more interframe independence");
	updateParameter("Iterations", int(10000))->SetTooltip("Max number of iterations to perform");
	updateParameter("Epsilon", double(5e-5));
	updateParameter("Stop K", int(10))->SetTooltip("How often epsilon should be evaluated and checked");
	updateParameter("Adapt Params", false)->SetTooltip("If true: lambda and alpha will be adapted so that the solution will look more or less the same, for one and the same input image and for different scalings.");
	updateParameter("Weight", false)->SetTooltip("If true: The regularizer will be adjust to smooth less at pixels with high edge probability");
	updateParameter("Overlay edges", false);
	updateParameter("K segments", int(10));
	updateParameter("KMeans iterations", int(5));
	updateParameter("KMeans epsilon", 1e-5);
	solver.reset(new Solver());
}
cv::cuda::GpuMat SLaT::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	// First we apply the mumford and shah algorithm to smooth the input image
	img.download(imageBuffer, stream);
	
	Par param;
	param.lambda = *getParameter<double>(0)->Data();
	param.alpha = *getParameter<double>(1)->Data();
	param.temporal = *getParameter<double>(2)->Data();
	param.iterations = *getParameter<int>(3)->Data();
	param.stop_eps = *getParameter<double>(4)->Data();
	param.stop_k = *getParameter<int>(5)->Data();
	param.adapt_params = *getParameter<bool>(6)->Data();
	param.weight = *getParameter<bool>(7)->Data();
	param.edges = *getParameter<bool>(8)->Data();
	stream.waitForCompletion();
	cv::Mat result = solver->run(imageBuffer.createMatHeader(), param);

	int rows = img.size().area();
	cv::cvtColor(result, lab, cv::COLOR_BGR2Lab);
	lab.convertTo(lab_32f, CV_32F);
	result.convertTo(smoothed_32f, CV_32F);
	tensor.create(rows, 6, CV_32F);
	smoothed_32f.reshape(1,rows).copyTo(tensor(cv::Range(), cv::Range(0, 3)));
	lab_32f.reshape(1,rows).copyTo(tensor(cv::Range(), cv::Range(3, 6)));
	
    cv::kmeans(tensor, *getParameter<int>(9)->Data(), labels,
		cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, *getParameter<int>(10)->Data(), *getParameter<double>(11)->Data()),
		1, cv::KMEANS_RANDOM_CENTERS, centers);

	labels = labels.reshape(1, img.rows);
	updateParameter("Labels", labels);
	updateParameter("Centers", centers); 
	return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(OtsuThreshold, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentMOG2, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentGrabCut, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentWatershed, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentKMeans, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ManualMask, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentMeanShift, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentCPMC, Image, Processing, Segmentation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(SLaT, Image, Processing, Segmentation)

