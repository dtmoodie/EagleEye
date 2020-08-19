#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include <Aquila/rcc/external_includes/cv_cudacodec.hpp>
#include <Aquila/rcc/external_includes/cv_videoio.hpp>

#include "MetaObject/core/detail/ConcurrentQueue.hpp"
#include "MetaObject/thread/ThreadHandle.hpp"
#include "MetaObject/thread/ThreadPool.hpp"

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#include "../OpenCVCudaNode.hpp"

namespace cv
{
    namespace cudacodec
    {
        class VideoWriter;
    }
} // namespace cv

namespace aqcore
{

    class VideoWriter : public OpenCVCudaNode
    {
      public:
        ~VideoWriter();
        MO_DERIVE(VideoWriter, OpenCVCudaNode)
            INPUT(aq::SyncedImage, image)

            PARAM(mo::EnumParam, codec, mo::EnumParam())
            PARAM(mo::WriteDirectory, outdir, {})
            PARAM(mo::WriteFile, filename, mo::WriteFile("video.avi"))
            PARAM(bool, using_gpu_writer, true)

            PARAM(bool, write_metadata, false)
            PARAM(std::string, metadata_stem, "metadata")
            PARAM(std::string, dataset_name, "")
            MO_SLOT(void, write_out)

        MO_END;

        void nodeInit(bool firstInit);

      protected:
        bool processImpl();
        boost::thread _write_thread;

        cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
        cv::Ptr<cv::VideoWriter> h_writer;

        struct WriteData
        {
            cv::Mat img;
            boost::optional<mo::Header> header;
        };
        moodycamel::ConcurrentQueue<WriteData> _write_queue;
    };

} // namespace aqcore
