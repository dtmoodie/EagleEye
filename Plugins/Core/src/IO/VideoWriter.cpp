#include "VideoWriter.h"
#include <boost/filesystem.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;



bool VideoWriter::ProcessImpl()
{
    if(h_writer == nullptr && d_writer == nullptr)
    {
        if (boost::filesystem::exists(filename.string()))
        {
            LOG(info) << "File exists, overwriting";
        }
        // Attempt to initialize the device writer first
        cv::cudacodec::EncoderParams params;
        try
        {
            d_writer = cv::cudacodec::createVideoWriter(filename.string(), image->GetSize(), 30, params);
        }catch(...)
        {
            using_gpu_writer_param.UpdateData(false);
        }
        if(using_gpu_writer)
        {
            h_writer.reset(new cv::VideoWriter);
            if(!h_writer->open(filename.string(), codec.getValue(), 30, image->GetSize(), image->GetChannels() == 3))
            {
                LOG(debug) << "Failed to open video writer with codec " << codec.getEnum() << " falling back on defaults";
                if(!h_writer->open(filename.string(), -1, 30, image->GetSize(), image->GetChannels() == 3))
                {
                    LOG(warning) << "Unable to fallback on default video writer parameters";
                }
            }
        }
    }
    if(d_writer)
    {
        d_writer->write(image->GetGpuMat(*_ctx->stream));
    }
    if(h_writer)
    {
        if(image->GetSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            h_writer->write(image->GetMat(*_ctx->stream));
        }else
        {
            cv::Mat mat = image->GetMat(*_ctx->stream);
            cuda::enqueue_callback_async([mat, this]()->void
            {
                this->h_writer->write(mat);
            }, *_ctx->stream);
        }
    }
    return true;
}



MO_REGISTER_CLASS(VideoWriter);
