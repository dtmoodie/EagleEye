#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_cudacodec.hpp>
#include <Aquila/rcc/external_includes/cv_videoio.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include "MetaObject/core/detail/ConcurrentQueue.hpp"
#include "MetaObject/thread/ThreadHandle.hpp"
#include "MetaObject/thread/ThreadPool.hpp"

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace cv
{
namespace cudacodec
{
class VideoWriter;
}
}

namespace aq
{
namespace nodes
{

class VideoWriter : public Node
{
  public:
    ~VideoWriter();
    MO_DERIVE(VideoWriter, Node)
        INPUT(SyncedMemory, image, nullptr)
        STATE(cv::Ptr<cv::cudacodec::VideoWriter>, d_writer, cv::Ptr<cv::cudacodec::VideoWriter>())
        STATE(cv::Ptr<cv::VideoWriter>, h_writer, cv::Ptr<cv::VideoWriter>())
        PARAM(mo::EnumParam, codec, mo::EnumParam())
        PARAM(mo::WriteDirectory, outdir, {})
        PARAM(mo::WriteFile, filename, mo::WriteFile("video.avi"))
        PARAM(bool, using_gpu_writer, true)
        MO_SLOT(void, write_out)
        PARAM(bool, write_metadata, false)
        PARAM(std::string, metadata_stem, "metadata")
        PARAM(std::string, dataset_name, "")
    MO_END;
    void nodeInit(bool firstInit);

  protected:
    bool processImpl();
    boost::thread _write_thread;
    struct WriteData
    {
        cv::Mat img;
        boost::optional<mo::Time_t> ts;
        size_t fn;
    };
    moodycamel::ConcurrentQueue<WriteData> _write_queue;
};
}
}
