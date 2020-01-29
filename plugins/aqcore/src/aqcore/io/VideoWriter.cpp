#include "VideoWriter.h"
#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/utilities/cuda/CudaCallbacks.hpp>
#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/thread/boost_thread.hpp>
#include <boost/filesystem.hpp>
#include <fstream>

using namespace aq;
using namespace aq::nodes;

VideoWriter::~VideoWriter()
{
    _write_thread.interrupt();
    _write_thread.join();
}

void VideoWriter::nodeInit(bool firstInit)
{
    _write_thread = boost::thread([this]() {
        size_t video_frame_number = 0;
        std::unique_ptr<std::ofstream> ofs;
        mo::setThisThreadName("VideoWriter");
        while (!boost::this_thread::interruption_requested())
        {
            WriteData data;
            if (_write_queue.try_dequeue(data) && h_writer)
            {
                mo::scoped_profile profile("Writing video");
                h_writer->write(data.img);
                if (!ofs && write_metadata)
                {
                    ofs.reset(new std::ofstream(outdir.string() + "/" + metadata_stem + ".txt"));
                    (*ofs) << dataset_name << std::endl;
                }
                if (ofs)
                {
                    (*ofs) << video_frame_number << " " << data.fn;
                    if (data.ts)
                        (*ofs) << " " << *data.ts;
                    (*ofs) << std::endl;
                }
                ++video_frame_number;
            }
            else
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(5));
            }
        }
    });
}

bool VideoWriter::processImpl()
{
    if (image->empty())
        return false;
    if (h_writer == nullptr && d_writer == nullptr)
    {
        if (!boost::filesystem::exists(outdir))
        {
            boost::system::error_code ec;
            boost::filesystem::create_directories(outdir, ec);
            if (ec)
            {
                MO_LOG(warning) << "Unable to create directory '" << outdir << "' " << ec.message();
            }
        }
        if (boost::filesystem::exists(outdir.string() + "/" + filename.string()))
        {
            MO_LOG(info) << "File exists, overwriting";
        }
// Attempt to initialize the device writer first
#if MO_OPENCV_HAVE_CUDA
        if (using_gpu_writer)
        {
            try
            {
                cv::cudacodec::EncoderParams params;
                d_writer = cv::cudacodec::createVideoWriter(
                    outdir.string() + "/" + filename.string(), image->getSize(), 30, params);
            }
            catch (...)
            {
                using_gpu_writer_param.updateData(false);
            }
        }
#endif
        if (!using_gpu_writer)
        {
            h_writer.reset(new cv::VideoWriter);
            if (!h_writer->open(outdir.string() + "/" + filename.string(),
                                cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
                                30,
                                image->getSize(),
                                image->getChannels() == 3))
            {
                MO_LOG(warning) << "Unable to open video writer for file " << filename;
            }
        }
    }
#if MO_OPENCV_HAVE_CUDA
    if (d_writer)
    {
        d_writer->write(image->getGpuMat(stream()));
    }
#endif
    if (h_writer)
    {
        cv::Mat h_img = image->getMat(stream());
        WriteData data;
        data.img = h_img;
        data.fn = image_param.getFrameNumber();
        data.ts = image_param.getTimestamp();
        cuda::enqueue_callback([data, this]() { _write_queue.enqueue(data); }, stream());
    }
    return true;
}

void VideoWriter::write_out()
{
    d_writer.release();
    h_writer.release();
}

MO_REGISTER_CLASS(VideoWriter)
