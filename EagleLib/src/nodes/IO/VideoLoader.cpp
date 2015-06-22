#include "nodes/IO/VideoLoader.h"
#if _WIN32
#include <opencv2/cudacodec.hpp>
#else
#include <opencv2/videoio.hpp>
#endif
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <ISimpleSerializer.h>
using namespace EagleLib;
//#define GPU_DECODE_ENABLED

VideoLoader::~VideoLoader()
{

}

void 
VideoLoader::Init(bool firstInit)
{
    Node::Init(firstInit);
	nodeType = eSource;
    if(firstInit)
    {
        updateParameter<boost::filesystem::path>("Filename", boost::filesystem::path("/home/dmoodie/Downloads/trailer.mp4"), Parameter::Control, "Path to video file");
        updateParameter<cv::Ptr<cv::cudacodec::VideoReader>>("GPU video reader", d_videoReader, Parameter::Output);
        updateParameter<cv::Ptr<cv::VideoCapture>>("CPU video reader", h_videoReader, Parameter::Output);
        updateParameter<boost::function<void(void)>>("Restart Video",boost::bind(&VideoLoader::restartVideo,this), Parameter::Control);
        updateParameter<bool>("Loop",true);
        updateParameter<bool>("End of video", false, Parameter::Output);
        load = false;
    }
    h_img.resize(20);
}
void VideoLoader::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(d_videoReader);
    SERIALIZE(h_videoReader);
}

cv::cuda::GpuMat 
VideoLoader::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    bool firstLoad = false;
    if (parameters[0]->changed || load)
    {
		loadFile();
        firstLoad = true;
        load = false;
    }
    if(d_videoReader)
    {
        d_videoReader->nextFrame(img);
    }else if(h_videoReader)
    {
        auto buffer = h_img.getFront();
        if(!h_videoReader->read(*buffer))
        {
            updateParameter<bool>(5, true);
            log(Status, "End of video reached");
            auto reload = getParameter<bool>("Loop");
            if(reload)
                if(reload->data)
                    loadFile();
            return img;
        }
       if(buffer->empty())
           return cv::cuda::GpuMat();
       try
       {
           img.upload(*buffer, stream);
       }catch(cv::Exception &err)
       {
           log(Error, err.what());
           return img;
       }
       updateParameter<double>("Timestamp",h_videoReader->get(cv::CAP_PROP_POS_MSEC), Parameter::State);
       updateParameter<int>("Frame index",(int)h_videoReader->get(cv::CAP_PROP_POS_FRAMES), Parameter::Output);
       updateParameter<double>("% Complete",h_videoReader->get(cv::CAP_PROP_POS_AVI_RATIO), Parameter::State);
       updateParameter("Source Image", img, Parameter::Output);
    }
    if(firstLoad && !img.empty())
    {
        std::stringstream ss;
        ss << "File loaded successfully! Resolution: " << img.size() << " channels: " << img.channels();
        log(Status,  ss.str());
    }
    return img;
}

void
VideoLoader::loadFile()
{

    auto fileName = getParameter<boost::filesystem::path>("Filename");
    if(fileName == nullptr)
        return;
    log(Status, "Loading file: " + fileName->data.string());

    if(!boost::filesystem::exists(fileName->data))
    {
        log(Warning, fileName->data.string() + " doesn't exist");
        return;
    }
#ifdef GPU_DECODE_ENABLED
    try
    {
        d_videoReader = cv::cudacodec::createVideoReader(fileName->data.string());
    }catch(...)
#endif
    {
        // no luck with the GPU decoder, try CPU decoder
        log(Error, "Failed to create GPU decoder, falling back to CPU decoder");
        h_videoReader.reset(new cv::VideoCapture);
        try
        {
            h_videoReader->open(fileName->data.string());
        }catch(cv::Exception &e)
        {
            log(Error, "Failed to fallback on CPU decoder");
        }

        /*
        -   **CV_CAP_PROP_POS_MSEC** Current position of the video file in milliseconds or video
            capture timestamp.
        -   **CV_CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next.
        -   **CV_CAP_PROP_POS_AVI_RATIO** Relative position of the video file: 0 - start of the
            film, 1 - end of the film.
        -   **CV_CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream.
        -   **CV_CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream.
        -   **CV_CAP_PROP_FPS** Frame rate.
        -   **CV_CAP_PROP_FOURCC** 4-character code of codec.
        -   **CV_CAP_PROP_FRAME_COUNT** Number of frames in the video file.
        -   **CV_CAP_PROP_FORMAT** Format of the Mat objects returned by retrieve() .
        -   **CV_CAP_PROP_MODE** Backend-specific value indicating the current capture mode.
        -   **CV_CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras).
        -   **CV_CAP_PROP_CONTRAST** Contrast of the image (only for cameras).
        -   **CV_CAP_PROP_SATURATION** Saturation of the image (only for cameras).
        -   **CV_CAP_PROP_HUE** Hue of the image (only for cameras).
        -   **CV_CAP_PROP_GAIN** Gain of the image (only for cameras).
        -   **CV_CAP_PROP_EXPOSURE** Exposure (only for cameras).
        -   **CV_CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted
            to RGB.
        -   **CV_CAP_PROP_WHITE_BALANCE** Currently not supported
        -   **CV_CAP_PROP_RECTIFICATION** Rectification flag for stereo cameras (note: only supported
            by DC1394 v 2.x backend currently)*/
        updateParameter<double>("Timestamp",h_videoReader->get(cv::CAP_PROP_POS_MSEC));
        updateParameter<int>("Frame index",(int)h_videoReader->get(cv::CAP_PROP_POS_FRAMES), Parameter::Output);
        updateParameter<double>("% Complete",h_videoReader->get(cv::CAP_PROP_POS_AVI_RATIO));
        updateParameter<double>("Frame count",h_videoReader->get(cv::CAP_PROP_FRAME_COUNT), Parameter::Output);

    }

    if (d_videoReader)
	{
        auto info = d_videoReader->format();
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
		
        std::string resolution = "Width: " + boost::lexical_cast<std::string>(info.width) +
                " Height: " + boost::lexical_cast<std::string>(info.height);
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
    fileName->changed = false;
    updateParameter<bool>(5, false);

}
bool VideoLoader::SkipEmpty() const
{
    return false;
}
void
VideoLoader::restartVideo()
{
    load = true;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(VideoLoader)
