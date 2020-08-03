#pragma once

#include "Aquila/utilities/cuda/CudaUtils.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#ifndef TOKEN_TO_STRING
#define TOKEN_TO_STRING(TOKEN) #TOKEN
#endif
#define ENUM_(TOKEN)                                                                                                   \
    {                                                                                                                  \
        TOKEN_TO_STRING(TOKEN), TOKEN                                                                                  \
    }

namespace aq
{
    namespace nodes
    {
        class PlaybackInfo : public Node
        {
          public:
            MO_DERIVE(PlaybackInfo, Node)
                INPUT(SyncedImage, input)
                STATUS(double, framerate, 0.0)
                STATUS(double, source_framerate, 0.0)
                STATUS(double, playrate, 0.0)
            MO_END;

          protected:
            bool processImpl() override;

          private:
            boost::posix_time::ptime last_iteration_time;
            boost::optional<mo::Time_t> last_timestamp;
        };

        class ImageInfo : public Node
        {
          public:
            MO_DERIVE(ImageInfo, Node)
                INPUT(SyncedImage, input)
                ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
                APPEND_FLAGS(data_type, mo::ParamFlags::State_e)
                STATUS(int, count, 0)
                STATUS(int, height, 0)
                STATUS(int, width, 0)
                STATUS(int, channels, 0)
                STATUS(int, ref_count, 0)
            MO_END;

          protected:
            bool processImpl() override;
        };

        class Mat2Tensor : public Node
        {
          public:
            MO_DERIVE(Mat2Tensor, Node)
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, output)
                ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
                PARAM(bool, include_position, true)
            MO_END
          protected:
            bool processImpl() override;

          private:
            cv::cuda::GpuMat position_mat;
        };

        class ConcatTensor : public Node
        {
            BufferPool<cv::cuda::GpuMat, EventPolicy> d_buffer;

          public:
            ConcatTensor();
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
            virtual void nodeInit(bool firstInit);
        };

        class LagBuffer : public Node
        {
            std::vector<cv::cuda::GpuMat> imageBuffer;
            unsigned int putItr;
            unsigned int getItr;
            unsigned int lagFrames;

          public:
            LagBuffer();
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
            virtual void nodeInit(bool firstInit);
        };

        class CameraSync : public Node
        {
          public:
            CameraSync();
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
            virtual void nodeInit(bool firstInit);
            bool SkipEmpty() const;
        };
    } // namespace nodes
} // namespace aq
