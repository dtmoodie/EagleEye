#pragma once
#include "nodes/Node.h"
#include <opencv2/videoio.hpp>
namespace EagleLib
{
    class Camera: public Node
    {
        virtual void changeStream(const std::string& gstreamParams);
        virtual void changeStream(int device);
        cv::VideoCapture cam;
        cv::cuda::HostMem hostBuffer;
    public:
        Camera();
        virtual void Init(bool firstInit);
        virtual bool SkipEmpty() const;
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream);
    };
}
