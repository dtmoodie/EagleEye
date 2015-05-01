#pragma once
#include <nodes/Node.h>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>

namespace EagleLib
{
    class VideoWriter : public Node
    {
    public:
        VideoWriter();
        VideoWriter(std::string fileName);
        void Init(bool firstInit);
        void Serialize(ISimpleSerializer *pSerializer);
        ~VideoWriter();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
        void writeImg(cv::cuda::GpuMat& img);
        void startWrite();
        void restartFunc();
        bool gpuWriter;
        bool restart;
        cv::Size size;
        cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
        cv::Ptr<cv::VideoWriter>    h_writer;
    };

}
