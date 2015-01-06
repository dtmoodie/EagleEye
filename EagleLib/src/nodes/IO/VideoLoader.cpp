#include "nodes/IO/VideoLoader.h"
#include <opencv2/cudacodec.hpp>

using namespace EagleLib;


IO::VideoLoader::VideoLoader()
{
	addParameter("VideoFileReader", cv::Ptr<cv::cudacodec::VideoReader>(), std::string("Object that decodes video files on the GPU"), Parameter::Output);
	addParameter("VideoFileName", std::string(""), std::string("Absolute file path to video file", Parameter::Control));
}

IO::VideoLoader::VideoLoader(std::string file)
{
	VideoLoader();
	updateParameter(1, file);
	loadFile();
}

IO::VideoLoader::~VideoLoader()
{

}

cv::cuda::GpuMat 
IO::VideoLoader::doProcess(cv::cuda::GpuMat& img)
{
	if (parameters[1]->changed)
		loadFile();
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
		EOF_reached = true;
	}
}
void
IO::VideoLoader::loadFile()
{
	auto ptr = boost::dynamic_pointer_cast<TypedParameter<cv::Ptr<cv::cudacodec::VideoReader>>, Parameter>(parameters[0]);
	if (ptr == NULL)
		return;
	auto fileName = getParameter<std::string>(1);
	if (fileName == NULL)
		return;
	if (fileName->data.size() > 0)
		ptr->data = cv::cudacodec::createVideoReader(fileName->data);
}