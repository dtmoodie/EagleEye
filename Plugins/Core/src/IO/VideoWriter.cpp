#include "VideoWriter.h"
#include <boost/filesystem.hpp>
#include "MetaObject/Logging/Profiling.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;


void VideoWriter::NodeInit(bool firstInit)
{

}
bool VideoWriter::ProcessImpl()
{
    if(image->empty())
        return false;
    if(h_writer == nullptr && d_writer == nullptr)
    {
        if (boost::filesystem::exists(filename.string()))
        {
            LOG(info) << "File exists, overwriting";
        }
        // Attempt to initialize the device writer first
        if(using_gpu_writer)
        {
            try
            {
                cv::cudacodec::EncoderParams params;
                d_writer = cv::cudacodec::createVideoWriter(filename.string(), image->GetSize(), 30, params);
            }
            catch (...)
            {
                using_gpu_writer_param.UpdateData(false);
            }
        }
        
        if(!using_gpu_writer)
        {
            h_writer.reset(new cv::VideoWriter);
            if(!h_writer->open(filename.string(), cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 30, image->GetSize(), image->GetChannels() == 3))
            {
                LOG(warning) << "Unable to open video writer for file " << filename;
            }
        }
    }
    if(d_writer)
    {
        d_writer->write(image->GetGpuMat(Stream()));
    }
    if(h_writer)
    {
        cv::Mat h_img = image->GetMat(Stream());
        cuda::enqueue_callback_async([h_img, this]()
        {
            mo::scoped_profile profile("Writing video");
            h_writer->write(h_img);
        }, _write_thread.GetId(), Stream());

    }
    return true;
}

void VideoWriter::write_out()
{
    d_writer.release();
    h_writer.release();
}

MO_REGISTER_CLASS(VideoWriter);


#ifdef HAVE_FFMPEG




#endif
