#include <nodes/IO/VideoWriter.h>
#include <boost/filesystem.hpp>
#include <EagleLib/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;


VideoWriter::VideoWriter(std::string fileName)
{
    updateParameter(0, boost::filesystem::path(fileName));
}

VideoWriter::~VideoWriter()
{

}
void VideoWriter::Init(bool firstInit)
{
    Node::Init(firstInit);
	if (firstInit)
	{
		Parameters::EnumParameter param;
        param.addEnum(cv::VideoWriter::fourcc('X','2','6','4'), "X264");
        param.addEnum(cv::VideoWriter::fourcc('Y','U','V','9'), "YUV9");
        param.addEnum(cv::VideoWriter::fourcc('Y','U','Y','V'), "YUYV");
        param.addEnum(cv::VideoWriter::fourcc('M','J','P','G'), "MPJG");
        updateParameter("Codec", param);
		updateParameter("Filename", Parameters::WriteFile(""));
		writeOut = false;
	}
    updateParameter<boost::function<void(void)>>("Restart Functor", boost::bind(&VideoWriter::restartFunc, this));
	updateParameter<boost::function<void(void)>>("Stop Functor", boost::bind(&VideoWriter::endWrite, this));
    restart = false;

}
void VideoWriter::restartFunc()
{
    restart = true;
}
void VideoWriter::endWrite()
{
	writeOut = true;
}

void VideoWriter::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(d_writer);
    SERIALIZE(h_writer);
}

cv::cuda::GpuMat 
VideoWriter::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    size = img.size();
    if(_parameters[0]->changed || _parameters[1]->changed || restart)
        startWrite();
    if(d_writer != nullptr || h_writer != nullptr)
        writeImg(img);
	if (writeOut)
	{
		if (h_writer)
			h_writer.release();
		if (d_writer)
			d_writer.release();
		writeOut = false;
	}
	return img;
}
void 
VideoWriter::writeImg(cv::cuda::GpuMat& img)
{
    if(d_writer)
    {
        d_writer->write(img);
        return;
    }
    if(h_writer)
    {
        cv::Mat h_img(img);
        h_writer->write(h_img);
    }
}
void
VideoWriter::startWrite()
{
    //log(Status, "Starting write");
	NODE_LOG(info) << "Starting write";
    auto param = getParameter<Parameters::WriteFile>(1);
    if(param == nullptr)
        return;
	if (boost::filesystem::exists(param->Data()->string()))
	{
		NODE_LOG(info) << "File exists, overwriting";
	}
		
    if(h_writer == nullptr)
    {
        try
        {
            cv::cudacodec::EncoderParams params;
            d_writer = cv::cudacodec::createVideoWriter(param->Data()->string(), size, 30, params);
            //log(Status, "Using GPU encoder");
			NODE_LOG(info) << "Using GPU encoder";
        }catch(cv::Exception &e)
        {
            h_writer = cv::Ptr<cv::VideoWriter>(new cv::VideoWriter);
            auto codec = getParameter<Parameters::EnumParameter>(0);
            bool success;
            if(codec)
            {
                success = h_writer->open(param->Data()->string(),codec->Data()->currentSelection,30,size);
            }else
            {
                success = h_writer->open(param->Data()->string(), -1, 30, size);
            }
			if (success)
			{
				NODE_LOG(info) << "Using CPU encoder";
			}
			else
			{
				NODE_LOG(info) << "Unable to open file";
			}

        }
    }else
    {
        h_writer = cv::Ptr<cv::VideoWriter>(new cv::VideoWriter);
        auto codec = getParameter<Parameters::EnumParameter>(0);
        bool success;
        if(codec)
        {
            success = h_writer->open(param->Data()->string(),codec->Data()->currentSelection,30,size);
        }else
        {
            success = h_writer->open(param->Data()->string(), -1, 30, size);
        }
		if (success)
		{
			NODE_LOG(info) << "Using CPU encoder";
		}			
		else
		{
			NODE_LOG(info) << "Unable to open file";
		}
			
    }
    _parameters[0]->changed = false;
    _parameters[1]->changed = false;
    restart = false;

}


NODE_DEFAULT_CONSTRUCTOR_IMPL(VideoWriter, Image, Sink);
