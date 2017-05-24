#include "VideoWriter.h"
#include <boost/filesystem.hpp>
#include "MetaObject/Logging/Profiling.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
#include <fstream>

using namespace aq;
using namespace aq::Nodes;

VideoWriter::~VideoWriter()
{
    _write_thread.interrupt();
    _write_thread.join();
}

void VideoWriter::NodeInit(bool firstInit){
    _write_thread = boost::thread(
    [this](){
        size_t video_frame_number = 0;
        std::unique_ptr<std::ofstream> ofs;
        while(!boost::this_thread::interruption_requested()){
            WriteData data;
            if(_write_queue.try_dequeue(data) && h_writer){
                mo::scoped_profile profile("Writing video");
                h_writer->write(data.img);
                if(!ofs && write_metadata){
                    ofs.reset(new std::ofstream(outdir.string() + "/" + metadata_stem + ".txt"));
                    (*ofs) << dataset_name << std::endl;
                }
                if(ofs){
                    (*ofs) << video_frame_number << " " << data.fn;
                    if(data.ts)
                        (*ofs) << " " << *data.ts;
                    (*ofs) << std::endl;
                }
                ++video_frame_number;
            }else{
                boost::this_thread::sleep_for(boost::chrono::milliseconds(5));
            }
        }
    });
}

bool VideoWriter::ProcessImpl()
{
    if(image->empty())
        return false;
    if(h_writer == nullptr && d_writer == nullptr){
        if(!boost::filesystem::exists(outdir)){
            boost::filesystem::create_directories(outdir);
        }
        if (boost::filesystem::exists(outdir.string() + "/" + filename.string())){
            LOG(info) << "File exists, overwriting";
        }
        // Attempt to initialize the device writer first
        if(using_gpu_writer){
            try{
                cv::cudacodec::EncoderParams params;
                d_writer = cv::cudacodec::createVideoWriter(outdir.string() + "/" + filename.string(), image->GetSize(), 30, params);
            }catch (...){
                using_gpu_writer_param.UpdateData(false);
            }
        }

        if(!using_gpu_writer){
            h_writer.reset(new cv::VideoWriter);
            if(!h_writer->open(outdir.string() + "/" + filename.string(), cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 30,
                               image->GetSize(), image->GetChannels() == 3)){
                LOG(warning) << "Unable to open video writer for file " << filename;
            }
        }
    }
    if(d_writer){
        d_writer->write(image->getGpuMat(Stream()));
    }
    if(h_writer){
        cv::Mat h_img = image->getMat(Stream());
        WriteData data;
        data.img = h_img;
        data.fn = image_param.GetFrameNumber();
        data.ts = image_param.GetTimestamp();
        cuda::enqueue_callback([data, this](){
            _write_queue.enqueue(data);
        },  Stream());
    }
    return true;
}

void VideoWriter::write_out(){
    d_writer.release();
    h_writer.release();
}

MO_REGISTER_CLASS(VideoWriter)


#ifdef HAVE_FFMPEG




#endif
