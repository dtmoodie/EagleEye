#pragma once

#include "src/precompiled.hpp"
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#define TOKEN_TO_STRING(TOKEN) #TOKEN
#define ENUM_(TOKEN) {TOKEN_TO_STRING(TOKEN), TOKEN}

namespace EagleLib
{
    namespace Nodes
    {
        class ImageInfo: public Node
        {
        public:
            MO_DERIVE(ImageInfo, Node)
                INPUT(SyncedMemory, input, nullptr);
                ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F);
                APPEND_FLAGS(data_type, mo::State_e);
                STATUS(int, count, 0);
                STATUS(int, height, 0);
                STATUS(int, width, 0);
                STATUS(int, channels, 0);
                STATUS(int, ref_count, 0);
            MO_END;
        protected:
            bool ProcessImpl();
        };
        
        class Mat2Tensor: public Node
        {
        public:
            MO_DERIVE(Mat2Tensor, Node)
                INPUT(SyncedMemory, input, nullptr);
                OUTPUT(SyncedMemory, output, SyncedMemory());
                ENUM_PARAM(data_type, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F);
                PARAM(bool, include_position, true);
            MO_END;
        protected:
            bool ProcessImpl();
            cv::cuda::GpuMat position_mat;
        };
        class ConcatTensor: public Node
        {
            BufferPool<cv::cuda::GpuMat, EventPolicy> d_buffer;
        public:
            ConcatTensor();
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
            virtual void NodeInit(bool firstInit);
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
            virtual void NodeInit(bool firstInit);
        };

        class CameraSync : public Node
        {
        public:
            CameraSync();
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
            virtual void NodeInit(bool firstInit);
            bool SkipEmpty() const;
        };
    } // namespace nodes
}
