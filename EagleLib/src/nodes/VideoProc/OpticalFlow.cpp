#include "nodes/VideoProc/OpticalFlow.h"

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;


BroxOpticalFlow::BroxOpticalFlow()
{
	addParameter(std::string("broxOpticalFlow"), new cv::cuda::BroxOpticalFlow(0.1f,0.1f,1,10,10,10), std::string("Used for tracking dense optical flow"), Parameter::Output);
	addParameter(std::string("horizontalFlow"), cv::cuda::GpuMat(), std::string("Flow along X-Axis"), Parameter::Output);
	addParameter(std::string("verticalFlow"), cv::cuda::GpuMat(), std::string("Flow along Y-Axis"), Parameter::Output);
	addParameter(std::string("alpha"), 0.1f, std::string("Flow smoothness parameter"), Parameter::Control);
	addParameter(std::string("gamma"), 0.1f, std::string("Gradient Consistency Importance"), Parameter::Control);
	addParameter(std::string("scaleFactor"), 0.50f, std::string("Pyramid Scale Factor"), Parameter::Control);
	addParameter(std::string("innerIterations"), int(10), std::string("Number of lagged non-linearity iterations"), Parameter::Control);
	addParameter(std::string("outerIterations"), int(10), std::string("Number of warping iterations (number of pyramid levels)"), Parameter::Control);
	addParameter(std::string("solverIterations"), int(1000), std::string("Number of linear system solver iterations"), Parameter::Control);
}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img)
{
    if(parameters[3]->changed || parameters[4]->changed || parameters[5]->changed || parameters[6]->changed || parameters[7]->changed || parameters[8]->changed)
    {
        parameters[0].reset(new TypedParameter< cv::cuda::BroxOpticalFlow* >(std::string("broxOpticalFlow"),
                                                                             std::string("Used for tracking dense optical flow"),
                                                                             new cv::cuda::BroxOpticalFlow(getParameter<float>(3)->data, getParameter<float>(4)->data,
                                                                                                           getParameter<float>(5)->data, getParameter<int>(6)->data,
                                                                                                           getParameter<int>(7)->data,   getParameter<int>(8)->data),
                                                                             Parameter::Output));
    }
    // This needs to be set as the first frame
    cv::cuda::GpuMat grey;
    if(img.channels() != 1)
    {
        cv::cuda::cvtColor(img,grey,cv::COLOR_BGR2GRAY);
    }else
        grey = img;
    grey.convertTo(grey, CV_32FC1);
    if(prevFrame.empty())
    {
        prevFrame = grey;
        return img;
    }
    
	(*getParameter<cv::cuda::BroxOpticalFlow*>(0)->data)(prevFrame, grey, getParameter<cv::cuda::GpuMat>(1)->data, getParameter<cv::cuda::GpuMat>(2)->data);
    prevFrame = grey;
    if(cpuCallback || gpuCallback || drawResults)
    {
        cv::cuda::GpuMat flowX = getParameter<cv::cuda::GpuMat>(1)->data;
        cv::cuda::GpuMat flowY = getParameter<cv::cuda::GpuMat>(2)->data;
        for(int i = 0; i < flowX.cols; ++i)
        {

        }
    }

}

PyrLKOpticalFlow::PyrLKOpticalFlow()
{
	addParameter("pyrLykOpticalFlow", new cv::cuda::PyrLKOpticalFlow(), "Used for dense or sparse optical flow", Parameter::Output);
	addParameter("windowSize", cv::Size(5,5), "Search window size");
	addParameter("maxLevel", int(10), "Max level");
	addParameter("maxIterations", int(10), "Max iterations");
	addParameter("useInitialFlow", true, "Set to use initial flow if it exists for sparse flow estimation");
	addParameter("dense", false, "Set to perform dense optical flow");
	addParameter("initialPoints", cv::cuda::GpuMat(), "Initial points to track", Parameter::Input);
	addParameter("trackedPoints", cv::cuda::GpuMat(), "Tracked points in this image", Parameter::Output);
	addParameter("resetReference", false, "Flag to reset reference image to next input image");
	/* Publish functions */
    addParameter("resetFunctor", boost::bind(&PyrLKOpticalFlow::setReference, this), "Function for resetting the reference image", Parameter::Output);
    addParameter("sparseFunctor", boost::bind(&PyrLKOpticalFlow::sparse, this), "Function for applying sparse optical flow", Parameter::Output);
    addParameter("denseFunctor", boost::bind(&PyrLKOpticalFlow::dense,this), "Function for applying dense optical flow", Parameter::Output);
}

void
PyrLKOpticalFlow::getInputs()
{
    if(inputSelector)
    {

    }
}
cv::cuda::GpuMat 
PyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img)
{
	return img;
}

void 
PyrLKOpticalFlow::sparse(cv::cuda::GpuMat& img, cv::cuda::GpuMat& pts, cv::cuda::GpuMat& results, cv::cuda::GpuMat* error)
{
	cv::cuda::GpuMat grey;
	if(img.channels() != 1)
	{
		cv::cuda::cvtColor(img,grey,cv::COLOR_BGR2GRAY);
	}else
		grey = img;
	if(refImg.empty())
	{
		refImg = grey;
		return;
	}
	// If there are reference points, we're tracking relative to a set reference frame
	if(!refPts.empty())
	{
		getParameter<cv::cuda::PyrLKOpticalFlow*>(0)->data->useInitialFlow = getParameter<bool>(4)->data && !prevPts.empty();
		getParameter<cv::cuda::PyrLKOpticalFlow*>(0)->data->sparse(refImg, grey, refPts, prevPts, results, error);
	}else
	{
		cv::cuda::GpuMat keyPoints = getParameter<cv::cuda::GpuMat>(6)->data;
		if(keyPoints.empty())
		{
			if(warningCallback)
				warningCallback(std::string("Cannot perform sparse tracking without setting initial key points"));
			return;
		}
		getParameter<cv::cuda::PyrLKOpticalFlow*>(0)->data->useInitialFlow = getParameter<bool>(4)->data && !prevPts.empty();
		getParameter<cv::cuda::PyrLKOpticalFlow*>(0)->data->sparse(refImg, grey, keyPoints, prevPts, results, error);
		// We are not tracking relative to a reference frame, thus track relative to previous frame
		refImg = grey;
	}
	/* consider filtering points */
	pts = prevPts;
	
}

void 
PyrLKOpticalFlow::dense(cv::cuda::GpuMat& img, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v, cv::cuda::GpuMat* error)
{
	cv::cuda::GpuMat grey;
	if(img.channels() != 1)
	{
		cv::cuda::cvtColor(img,grey,cv::COLOR_BGR2GRAY);
	}else
		grey = img;
	if(refImg.empty())
	{
		refImg = grey;
		return;
	}
	getParameter<cv::cuda::PyrLKOpticalFlow*>(0)->data->dense(refImg, grey, u, v, error);
}
void 
PyrLKOpticalFlow::setReference(cv::cuda::GpuMat& img, cv::cuda::GpuMat* refPts_)
{
	if(img.channels() != 1)
	{
		cv::cuda::cvtColor(img,refImg,cv::COLOR_BGR2GRAY);
	}else
		refImg = img;
	if(refPts_ != NULL)
		refPts = *refPts_;



}
