#include "nodes/IO/VideoLoader.h"
#if _WIN32
#include <opencv2/cudacodec.hpp>
#else
#include <opencv2/videoio.hpp>
#endif

using namespace EagleLib;
using namespace EagleLib::IO;

VideoLoader::VideoLoader(const std::string& file)
{
    nodeName = "VideoLoader";
	treeName = nodeName;/*
#if _WIN32
	addParameter("VideoFileReader", cv::Ptr<cv::cudacodec::VideoReader>(), std::string("Object that decodes video files on the GPU"), Parameter::Output);
#else
    addParameter("VideoFileReader", cv::VideoCapture(), "Object that decodes video files on the GPU", Parameter::Output);
#endif
	addParameter("VideoFileName", std::string(""), std::string("Absolute file path to video file", Parameter::Control));
	addParameter("EOF_reached", false, "Flag for end of file", Parameter::Output);
	addParameter("NumFrames", int(-1), "Number of frames in file", Parameter::Output);
	updateParameter(1, file);*/
	loadFile();
}

VideoLoader::~VideoLoader()
{

}

cv::cuda::GpuMat 
VideoLoader::doProcess(cv::cuda::GpuMat& img)
{
	if (parameters[1]->changed)
		loadFile();
#if _WIN32
	auto ptr = boost::dynamic_pointer_cast<TypedParameter<cv::Ptr<cv::cudacodec::VideoReader>>, Parameter>(parameters[0]);
    if (ptr == NULL)
        return img;
    if (ptr->data == NULL)
        return img;
    if(ptr->data->nextFrame(img))
    {
        return img;
    }else
    {
        // Maybe this is the end of the video file?
        getParameter<bool>(2)->data = true;
        return img;
    }
#else
     TypedParameter<cv::VideoCapture>::Ptr ptr = boost::dynamic_pointer_cast<TypedParameter<cv::VideoCapture>, Parameter>(parameters[0]);
     if(!ptr->data.isOpened())
     {
        if(!ptr->data.open(getParameter<std::string>(1)->data))
            return img;
     }
     cv::Mat readFrame;
     if(!ptr->data.read(readFrame))
         return img;
     if(readFrame.empty())
         return img;
     img.upload(readFrame);
     return img;
#endif
}
void
VideoLoader::loadFile()
{
#if _WIN32
	auto ptr = boost::dynamic_pointer_cast<TypedParameter<cv::Ptr<cv::cudacodec::VideoReader>>, Parameter>(parameters[0]);
#else
    auto ptr = boost::dynamic_pointer_cast< TypedParameter<cv::VideoCapture> , Parameter>(parameters[0]);
#endif
	if (ptr == NULL)
		return;
	auto fileName = getParameter<std::string>(1);
	if (fileName == NULL)
		return;
	if (fileName->data.size() > 0)
    {
#if _WIN32
		EAGLE_TRY_ERROR(
        ptr->data = cv::cudacodec::createVideoReader(fileName->data);
		)
#else
		EAGLE_TRY_ERROR(
		ptr->data = cv::VideoCapture(fileName->data);
		)
#endif
        
    }

}
REGISTERCLASS(VideoLoader)