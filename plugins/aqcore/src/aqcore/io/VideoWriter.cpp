#include "VideoWriter.h"
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <boost/filesystem.hpp>

#include <fstream>

namespace aqcore
{

    VideoWriter::~VideoWriter()
    {
        _write_thread.interrupt();
        _write_thread.join();
    }

    void VideoWriter::nodeInit(bool )
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
                    PROFILE_RANGE(WritingVideo);
                    h_writer->write(data.img);
                    if (!ofs && write_metadata)
                    {
                        ofs.reset(new std::ofstream(outdir.string() + "/" + metadata_stem + ".txt"));
                        (*ofs) << dataset_name << std::endl;
                    }
                    if (ofs)
                    {
                        if (data.header)
                        {
                            (*ofs) << video_frame_number << " " << data.header->frame_number;
                            if (data.header->timestamp)
                            {
                                (*ofs) << " " << *data.header->timestamp;
                            }
                        }

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
        return false;
    }

    bool VideoWriter::processImpl(mo::IDeviceStream& stream)
    {
        return this->processImpl(static_cast<mo::IAsyncStream&>(stream));
    }

    bool VideoWriter::processImpl(mo::IAsyncStream& stream)
    {
        if (this->image->empty())
        {
            return false;
        }

        if (this->h_writer == nullptr && this->d_writer == nullptr)
        {
            if (!boost::filesystem::exists(outdir))
            {
                boost::system::error_code ec;
                boost::filesystem::create_directories(outdir, ec);
                if (ec)
                {
                    this->getLogger().warn("Unable to create directory {} due to {}", outdir, ec.message());
                }
            }
            if (boost::filesystem::exists(outdir.string() + "/" + filename.string()))
            {
                this->getLogger().info("File exists, overwriting");
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
            if (!this->using_gpu_writer)
            {
                const aq::Shape<3> shape = image->shape();
                const cv::Size size(shape(1), shape(0));
                const bool color = shape(2) == 3;
                const auto fourcc = cv::VideoWriter::fourcc('M', 'P', 'E', 'G');
                this->h_writer.reset(new cv::VideoWriter);
                const std::string file_name = outdir.string() + "/" + filename.string();
                if (!this->h_writer->open(file_name, fourcc, 30, size, color))
                {
                    this->getLogger().warn("Unable to open video writer for file {}", filename);
                }
            }
        }
#if MO_OPENCV_HAVE_CUDA
        if (d_writer)
        {
            d_writer->write(image->getGpuMat(stream()));
        }
#endif
        if (this->h_writer)
        {

            bool sync = false;
            cv::Mat h_img = image->getMat(&stream, &sync);

            WriteData data;
            data.img = h_img;
            data.header = image_param.getNewestHeader();
            auto work = [data, this](mo::IAsyncStream&) { _write_queue.enqueue(data); };
            if (sync)
            {
                stream.pushWork(std::move(work));
            }
            else
            {
                work(stream);
            }
        }
        return true;
    }

    void VideoWriter::write_out()
    {
        d_writer.release();
        h_writer.release();
    }

} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(VideoWriter)
