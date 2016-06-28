#pragma once
#include <EagleLib/nodes/Node.h>
#include <EagleLib/rcc/external_includes/cv_videoio.hpp>
#include <EagleLib/rcc/external_includes/cv_cudacodec.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class VideoWriter : public Node
    {
        bool gpuWriter;
        bool restart;
        cv::Size size;
        cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
        cv::Ptr<cv::VideoWriter>    h_writer;
        bool writeOut;
    public:
        VideoWriter();
        VideoWriter(std::string fileName);
        void NodeInit(bool firstInit);
        void Serialize(ISimpleSerializer *pSerializer);
        ~VideoWriter();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        void writeImg(cv::cuda::GpuMat& img);
        void startWrite();
        void restartFunc();
        void endWrite();
        
    };
    }
}
