#pragma once

#include "nodes/Node.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class ImageWriter: public Node
    {
        enum Extensions
        {
            jpg = 0,
            png,
            tiff,
            bmp
        };

        std::string baseName;
        std::string extension;
        int frameCount;
        bool writeRequested;
        cv::cuda::HostMem h_buf;
        int frameSkip;
    public:
        ImageWriter();
        void requestWrite();
        void writeImage();

        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
