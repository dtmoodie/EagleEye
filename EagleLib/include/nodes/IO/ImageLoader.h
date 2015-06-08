#pragma once
#include "nodes/Node.h"

namespace EagleLib
{

    class ImageLoader: public Node
    {
        cv::cuda::GpuMat d_img;
        void load();
    public:
        ImageLoader();
        virtual bool SkipEmpty() const;
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class DirectoryLoader: public Node
    {

        std::vector<std::string> files;
        int fileIdx;
    public:
        void restart();
        DirectoryLoader();
        virtual bool SkipEmpty() const;
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
