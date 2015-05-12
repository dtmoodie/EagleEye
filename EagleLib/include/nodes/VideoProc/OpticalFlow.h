#pragma once
#include <nodes/Node.h>
#include <Manager.h>
#include <opencv2/cudaoptflow.hpp>
#include "CudaUtils.hpp"
namespace EagleLib
{
    class SparsePyrLKOpticalFlow: public Node
    {
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlow;
        cv::cuda::GpuMat prevGreyImg;
        cv::cuda::GpuMat prevKeyPoints;
        cv::cuda::GpuMat refImg;
        cv::cuda::GpuMat refPts;
        cv::cuda::GpuMat trackedKeyPoints;
        cv::cuda::GpuMat status;
        cv::cuda::GpuMat error;

    public:
        SparsePyrLKOpticalFlow();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream = cv::cuda::Stream::Null());
        virtual cv::cuda::GpuMat track(cv::cuda::GpuMat img, cv::cuda::GpuMat* status = nullptr, cv::cuda::GpuMat* error = status);
        virtual int setReferenceImage(cv::cuda::GpuMat img, cv::cuda::GpuMat keyPoints, cv::cuda::Stream stream = cv::cuda::Stream::Null(), int idx = -1);
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
    public:
    };
}
