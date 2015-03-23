#include "nodes/IO/VideoLoader.h"
#if _WIN32
#include <opencv2/cudacodec.hpp>
#else
#include <opencv2/videoio.hpp>
#endif
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
using namespace EagleLib;
using namespace EagleLib::IO;

NODE_DEFAULT_CONSTRUCTOR_IMPL(VideoLoader);

VideoLoader::~VideoLoader()
{

}

void 
VideoLoader::Init(bool firstInit)
{
	updateParameter<boost::filesystem::path>("Filename", boost::filesystem::path(), Parameter::Control, "Path to video file");
	updateParameter<std::string>("Codec", "");
	updateParameter<std::string>("Video Chroma Format", "", Parameter::State);
	updateParameter<std::string>("Resolution", "", Parameter::State);
}



cv::cuda::GpuMat 
VideoLoader::doProcess(cv::cuda::GpuMat& img)
{
	if (parameters[0]->changed)
		loadFile();
#if _WIN32
	if (videoReader == NULL)
        return img;
    if(videoReader->nextFrame(img))
    {
        return img;
    }else
    {
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
	std::string& fileName = getParameter<std::string>("Filename")->data;
	if (fileName.size())
		videoReader = cv::cudacodec::createVideoReader(fileName);
	if (videoReader)
	{
		auto info = videoReader->format();
		std::string chromaFormat;
		switch (info.chromaFormat)
		{
		case cv::cudacodec::Monochrome:
			chromaFormat = "Monochrome";
			break;
		case cv::cudacodec::YUV420:
			chromaFormat = "YUV420";
			break;
		case cv::cudacodec::YUV422:
			chromaFormat = "YUV422";
			break;
		case cv::cudacodec::YUV444:
			chromaFormat = "YUV444";
			break;
		}
		
		std::string resolution = "Width: " + boost::lexical_cast<std::string>(info.width) + " Height: " + boost::lexical_cast<std::string>(info.height);
		
		std::string codec;
		switch (info.codec)
		{
		case cv::cudacodec::MPEG1:
			codec = "MPEG1";
			break;
		case cv::cudacodec::MPEG2:
			codec = "MPEG2";
			break;
		case cv::cudacodec::MPEG4:
			codec = "MPEG4";
			break;
		case cv::cudacodec::VC1:
			codec = "VC1";
			break;
		case cv::cudacodec::H264:
			codec = "H264";
			break;
		case cv::cudacodec::JPEG:
			codec = "JPEG";
			break;
		case cv::cudacodec::H264_SVC:
			codec = "H264_SVC";
			break;
		case cv::cudacodec::H264_MVC:
			codec = "H264_MVC";
			break;
		case cv::cudacodec::Uncompressed_YUV420:
			codec = "Uncompressed_YUV420";
			break;
		case cv::cudacodec::Uncompressed_YV12:
			codec = "Uncompressed_YV12";
			break;
		case cv::cudacodec::Uncompressed_NV12:
			codec = "Uncompressed_NV12";
			break;
		case cv::cudacodec::Uncompressed_YUYV:
			codec = "Uncompressed_YUYV";
			break;
		case cv::cudacodec::Uncompressed_UYVY:
			codec = "Uncompressed_UYVY";
			break;
		}
		
		updateParameter<std::string>("Codec", codec);
		updateParameter<std::string>("Video Chroma Format", chromaFormat, Parameter::State);
		updateParameter<std::string>("Resolution", resolution, Parameter::State);
	}
#else
    auto ptr = boost::dynamic_pointer_cast< TypedParameter<cv::VideoCapture> , Parameter>(parameters[0]);
#endif
	

}