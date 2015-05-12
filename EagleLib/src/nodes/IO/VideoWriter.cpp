#include <nodes/IO/VideoWriter.h>
using namespace EagleLib;



VideoWriter::VideoWriter(std::string fileName)
{
    updateParameter(0, boost::filesystem::path(fileName));
}

VideoWriter::~VideoWriter()
{

}
void VideoWriter::Init(bool firstInit)
{
	if (firstInit)
	{
        EnumParameter param;
        param.addEnum(cv::VideoWriter::fourcc('X','2','6','4'), "X264");
        param.addEnum(cv::VideoWriter::fourcc('Y','U','V','9'), "YUV9");
        param.addEnum(cv::VideoWriter::fourcc('Y','U','Y','V'), "YUYV");
        param.addEnum(cv::VideoWriter::fourcc('M','J','P','G'), "MPJG");
        updateParameter("Codec", param);
        updateParameter("Filename", boost::filesystem::path(""));
	}
    updateParameter<boost::function<void(void)>>("Restart Functor", boost::bind(&VideoWriter::restartFunc, this));
    restart = false;

}
void VideoWriter::restartFunc()
{
    restart = true;
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
    if(parameters[0]->changed || parameters[1]->changed || restart)
        startWrite();
    if(d_writer != nullptr || h_writer != nullptr)
        writeImg(img);
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
    log(Status, "Starting write");
    auto param = getParameter<boost::filesystem::path>(1);
    if(param == nullptr)
        return;
    if(boost::filesystem::exists(param->data))
        log(Warning, "File exists, overwriting");
    if(h_writer == nullptr)
    {
        try
        {
            cv::cudacodec::EncoderParams params;
            d_writer = cv::cudacodec::createVideoWriter(param->data.string(), size, 30, params);
            log(Status, "Using GPU encoder");
        }catch(cv::Exception &e)
        {
            h_writer = cv::Ptr<cv::VideoWriter>(new cv::VideoWriter);
            auto codec = getParameter<EnumParameter>(0);
            bool success;
            if(codec)
            {
                success = h_writer->open(param->data.string(),codec->data.currentSelection,30,size);
            }else
            {
                success = h_writer->open(param->data.string(), -1, 30, size);
            }
            if(success)
                log(Status, "Using CPU encoder");
            else
                log(Status, "Unable to open file");

        }
    }else
    {
        h_writer = cv::Ptr<cv::VideoWriter>(new cv::VideoWriter);
        auto codec = getParameter<EnumParameter>(0);
        bool success;
        if(codec)
        {
            success = h_writer->open(param->data.string(),codec->data.currentSelection,30,size);
        }else
        {
            success = h_writer->open(param->data.string(), -1, 30, size);
        }
        if(success)
            log(Status, "Using CPU encoder");
        else
            log(Status, "Unable to open file");
    }
    parameters[0]->changed = false;
    parameters[1]->changed = false;
    restart = false;

}

NODE_DEFAULT_CONSTRUCTOR_IMPL(VideoWriter);
