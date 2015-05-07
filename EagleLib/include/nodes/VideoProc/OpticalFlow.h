#pragma once
#include <nodes/Node.h>
#include <Manager.h>
#include <opencv2/cudaoptflow.hpp>

namespace EagleLib
{
    class SparsePyrLKOpticalFlow: public Node
    {
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlow;
        cv::cuda::GpuMat prevGreyImg;
        cv::cuda::GpuMat prevKeyPoints;

    public:
        SparsePyrLKOpticalFlow();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };

    class BroxOpticalFlow: public Node
    {
    public:
        BroxOpticalFlow();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
    };
    class SparseOpticalFlow: public Node
    {
        cv::Ptr<cv::cuda::SparseOpticalFlow> optFlow;
    public:

    };

    class DenseOpticalFlow: public Node
    {
        //cv::Ptr<cv::cuda::DenseOpticalFlow>
    public:
    };
}

//using namespace EagleLib;
//REGISTERCLASS(BroxOpticalFlow)
//REGISTERCLASS(PyrLKOpticalFlow)
